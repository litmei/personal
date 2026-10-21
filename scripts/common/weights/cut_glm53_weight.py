"""
有什么用？
对 GLM-5.3 W8A8 模型（modelslim 导出，arch=GlmMoeDsaForCausalLM）按层数裁剪，
用于小规模 bring-up / 精度对拍。

GLM-5.3 W8A8 权重的结构要点（与 GLM-5.2 的差异，脚本已按此适配）：
1. 主模型 num_hidden_layers=78，前 3 层 dense（first_k_dense_replace=3），
   其余为 sparse（MoE，256 experts）；MoE gate 与 e_score_correction_bias 是 FLOAT，
   专家/attention 线性层是 W8A8，带 weight / weight_scale / weight_offset 三件套。
2. MTP（nextn）层内联在同一份权重里，key 前缀是 model.layers.78.*
   （config: num_nextn_predict_layers=1；含 eh_proj / enorm / hnorm / embed_tokens /
   shared_head.head / shared_head.norm + 完整 MoE + self_attn）。
3. config.json 里有 indexer_types / mlp_layer_types 两个长度=层数的列表（DSA 索引器
   是否复用上层 topk、每层 dense/sparse），裁剪时必须同步截断。
4. quant_model_description.json 除权重条目外还有 model_quant_type / version /
   optional / is_rot_used / group_size / metadata 等非权重 key，原样保留。
5. 除主分片外还有 rot.safetensors（唯一 key=rot.weight，非 layer key），原样保留。

如何使用？
1. 按照下方“手动填写区域”的指示完成配置
2. 运行该脚本 python xx.py
3. SCRIPT_MODE 两种模式：
   - "init"  首次使用：从 SRC 全量构建 DST，并将本脚本自身拷贝到 DST
             （副本的模式会被改为 "update"）
   - "update" 修改层数：同样从 SRC 重新构建 DST（需要 SRC 可访问），
             用法：在 DST 目录下的脚本副本中改好 DST_LAYERS 后运行

为什么不能只用软链接 + 改 index？（重要）
sglang 加载权重时，是按“index 引用的每个分片文件”逐一打开、
迭代“文件内部的全部 key”（而不是 index weight_map 里的 key）；
w8a8 目录（无 model.safetensors.index.json）甚至连文件都不筛，
直接 glob 目录下全部 *.safetensors。
因此分片文件内容必须与裁剪后的层数自洽，否则：
- draft(nextn) 模型用前缀 model.layers.{num_hidden_layers} 识别 nextn 权重，
  裁剪到 N 层后该前缀会撞上文件里“真实的第 N 层”（主模型层，含量化参数），
  导致 nextn 加载错乱/报错（KeyError: fused_qkv_a_proj_with_mqa.weight_offset）；
- 真正的 nextn 层（GLM-5.3 原名 layer 78）不改名则永远匹配不上前缀。
所以本脚本的处理方式：
- 含 nextn 层的分片 → 重写文件，将其 layer id 改名为 DST_LAYERS
- 含真实第 DST_LAYERS 层的分片 → 重写文件，删除这些 key（避免前缀冲突）
- 其余分片 → 软链接（多余的被裁剪层 key 无害：主模型按 end_layer 跳过，
  draft 按前缀跳过）
- config.json / safetensors index / quant_model_description.json(若存在)
  均从 SRC 重新生成（每次全量重建，天然幂等）

依赖：
safetensors + torch（重写分片时使用，需在含这两个库的环境运行，
如 conda 环境 nb313）；其余仅 Python 标准库。
"""

# ==================== 手动填写区域 ====================
SRC_MODEL_PATH = r"/home/litmei/workspace/weights/glm53_w8a8"
DST_MODEL_PATH = r"/home/litmei/workspace/weights/glm53_w8a8_cut"
DST_LAYERS = 10
SCRIPT_MODE = "init"
# =====================================================

import json
import os
import re
import shutil
import struct
import sys

import safetensors
import safetensors.torch
import torch  # noqa: F401  (safetensors framework="pt" 依赖)

LAYER_PREFIX = "model.layers."
EXPECTED_ARCH = "GlmMoeDsaForCausalLM"
# 需要随层数同步截断的 config 列表字段（顺序无关，缺省则跳过）
CONFIG_LAYER_LISTS = ("mlp_layer_types", "indexer_types")


def fail(msg):
    print(f"[错误] {msg}")
    sys.exit(1)


def warn(msg):
    print(f"[警告] {msg}")


def parse_layer_id(key):
    # "model.layers.10.mlp.xxx" -> 10，非 layer 权重返回 None
    if not key.startswith(LAYER_PREFIX):
        return None
    seg = key[len(LAYER_PREFIX):].split(".", 1)[0]
    return int(seg) if seg.isdigit() else None


def find_index_path(model_dir):
    # w8a8: quant_model_weights.safetensors.index.json
    candidates = sorted(n for n in os.listdir(model_dir) if n.endswith(".safetensors.index.json"))
    if len(candidates) != 1:
        fail(f"在 {model_dir} 下应恰好有一个 *.safetensors.index.json，实际找到: {candidates}")
    return os.path.join(model_dir, candidates[0])


def classify(key, orig_layers, dst_layers):
    """返回裁剪后该 key 的新名字；返回 None 表示该 key 应删除。
    - 非 layer 权重与 layer id < dst_layers：保留原名
    - layer id == orig_layers（nextn 层）：改名为 layer dst_layers
    - layer id 在 [dst_layers, orig_layers)：删除
    """
    lid = parse_layer_id(key)
    if lid is None or lid < dst_layers:
        return key
    if lid == orig_layers:
        return LAYER_PREFIX + str(dst_layers) + key[len(LAYER_PREFIX) + len(str(lid)):]
    return None


def read_st_header(path):
    """读取 safetensors 文件头，返回 (头部长度, 头部dict)"""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise ValueError("文件过小，不是有效的 safetensors")
        hlen = struct.unpack("<Q", raw)[0]
        header = json.loads(f.read(hlen).decode("utf-8"))
    return hlen, header


def rewrite_st_file(src_path, dst_path, keep):
    """把 src_path 中 keep={旧名:新名} 的张量写入 dst_path（其余丢弃）。
    用 safetensors 库读写，dtype/shape/metadata 原样保留。"""
    with safetensors.safe_open(src_path, framework="pt", device="cpu") as f:
        tensors = {new: f.get_tensor(old) for old, new in keep.items()}
        metadata = f.metadata()
    safetensors.torch.save_file(tensors, dst_path, metadata=metadata)


def build_cut():
    """init/update 共用：从 SRC 全量重建 DST"""
    # step0: 基础校验
    if not os.path.isdir(SRC_MODEL_PATH):
        fail(f"SRC_MODEL_PATH 不存在: {SRC_MODEL_PATH}")
    src_abs = os.path.abspath(SRC_MODEL_PATH)
    dst_abs = os.path.abspath(DST_MODEL_PATH)
    if dst_abs == src_abs or dst_abs.startswith(src_abs + os.sep) or src_abs.startswith(dst_abs + os.sep):
        fail("SRC_MODEL_PATH 与 DST_MODEL_PATH 不能相同或互相嵌套")
    if not isinstance(DST_LAYERS, int) or isinstance(DST_LAYERS, bool) or DST_LAYERS <= 0:
        fail(f"DST_LAYERS 必须是正整数，当前为: {DST_LAYERS!r}")

    src_config_path = os.path.join(src_abs, "config.json")
    if not os.path.isfile(src_config_path):
        fail(f"缺少文件: {src_config_path}")
    with open(src_config_path, encoding="utf-8") as f:
        config = json.load(f)
    arch = (config.get("architectures") or [""])[0]
    if arch != EXPECTED_ARCH:
        fail(f"该脚本只适用于 {EXPECTED_ARCH}，当前 config.architectures={config.get('architectures')}")
    orig_layers = config["num_hidden_layers"]
    if DST_LAYERS > orig_layers:
        fail(f"DST_LAYERS({DST_LAYERS}) 不能大于原始层数 num_hidden_layers({orig_layers})")

    index_path = find_index_path(src_abs)
    index_name = os.path.basename(index_path)
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # GLM-5.3 的 MTP 层内联在 layer {orig_layers}；缺失时无法作为 EAGLE/MTP 草稿使用
    mtp_probe = f"{LAYER_PREFIX}{orig_layers}.eh_proj.weight"
    has_mtp = mtp_probe in weight_map
    src_mtp_keys = [k for k in weight_map if parse_layer_id(k) == orig_layers]
    if not has_mtp:
        warn(
            f"未找到 {mtp_probe}：该权重似乎没有内联 MTP(nextn) 层，"
            "裁剪后不能作为 EAGLE/MTP 草稿模型使用（主模型不受影响）"
        )

    # dense 层数：DST_LAYERS 太小时模型会退化成纯 dense，确认是有意为之
    first_k_dense = config.get("first_k_dense_replace", 0)
    if DST_LAYERS <= first_k_dense:
        warn(f"DST_LAYERS({DST_LAYERS}) <= first_k_dense_replace({first_k_dense})，裁剪后没有任何 MoE 层")

    for field in CONFIG_LAYER_LISTS:
        vals = config.get(field)
        if isinstance(vals, list) and len(vals) != orig_layers:
            warn(f"config.{field} 长度 {len(vals)} 与 num_hidden_layers({orig_layers}) 不一致，仍按其自身截断")

    desc_name = "quant_model_description.json"
    has_desc = os.path.isfile(os.path.join(src_abs, desc_name))
    if not has_desc:
        warn(f"缺少 {desc_name}，按非 w8a8 目录处理（不重建量化描述）")

    # step1: mkdir -p DST
    os.makedirs(dst_abs, exist_ok=True)

    # step2: 清理 DST 中的旧产物（分片/旧版 .bak），随后全量重建
    for name in os.listdir(dst_abs):
        p = os.path.join(dst_abs, name)
        if name.endswith(".safetensors") or name.endswith(".bak"):
            if os.path.isfile(p) or os.path.islink(p):
                os.remove(p)

    # step3: 逐个处理 SRC 下的 safetensors 分片
    #   - 含 nextn 层(layer orig)或真实第 dst 层 → 重写（改名/删除冲突 key）
    #   - 仅含保留层/无害多余层 → 软链接
    #   - 全部是被裁剪层且无冲突 → 不生成
    created_files = set()
    total_size = 0
    n_symlink = n_rewrite = n_skip = 0
    for name in sorted(os.listdir(src_abs)):
        src_file = os.path.join(src_abs, name)
        if not name.endswith(".safetensors") or not os.path.isfile(src_file):
            continue
        dst_file = os.path.join(dst_abs, name)
        try:
            _, header = read_st_header(src_file)
        except Exception:
            # 无法解析头部的文件（如占位文件）原样软链
            os.symlink(src_file, dst_file)
            created_files.add(name)
            n_symlink += 1
            print(f"[step3] 软链接 {name}（头部无法解析，原样保留）")
            continue
        header.pop("__metadata__", None)
        keep = {}
        must_rewrite = False
        for k, info in header.items():
            lid = parse_layer_id(k)
            if DST_LAYERS < orig_layers and lid in (DST_LAYERS, orig_layers):
                must_rewrite = True  # 真实第 dst 层（前缀冲突）或 nextn 层（需改名）
            nk = classify(k, orig_layers, DST_LAYERS)
            if nk is not None:
                keep[k] = nk
                total_size += info["data_offsets"][1] - info["data_offsets"][0]
        if must_rewrite and keep:
            rewrite_st_file(src_file, dst_file, keep)
            created_files.add(name)
            n_rewrite += 1
            n_renamed = sum(1 for old, new in keep.items() if old != new)
            print(f"[step3] 重写   {name}（保留 {len(keep)} 项，删除 {len(header) - len(keep)} 项，nextn 改名 {n_renamed} 项）")
        elif keep:
            os.symlink(src_file, dst_file)
            created_files.add(name)
            n_symlink += 1
            print(f"[step3] 软链接 {name}")
        else:
            n_skip += 1
            print(f"[step3] 跳过   {name}（全部为被裁剪层）")
    print(f"[step3] 汇总: 重写 {n_rewrite}，软链 {n_symlink}，跳过 {n_skip}")

    missing = sorted(f for f in set(weight_map.values()) if not os.path.isfile(os.path.join(src_abs, f)))
    if missing:
        print(f"[警告] SRC 中有 {len(missing)} 个 index 引用的分片文件不存在（结构测试目录?），其 index 映射将原样保留")

    # step4: 重新生成 safetensors index（key 裁剪/nextn 改名，文件名不变）
    new_map = {}
    for k, fname in weight_map.items():
        nk = classify(k, orig_layers, DST_LAYERS)
        if nk is None:
            continue
        if fname in created_files or not os.path.isfile(os.path.join(src_abs, fname)):
            new_map[nk] = fname
    index["weight_map"] = new_map
    index.setdefault("metadata", {})["total_size"] = total_size
    index_dst = os.path.join(dst_abs, index_name)
    with open(index_dst, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    print(f"[step4] index: 剩余 {len(new_map)} 项, total_size={total_size}")

    # step4.1: 自检——不允许残留 layer id > DST_LAYERS 的 key；MTP 改名数量要对得上
    bad = sorted({lid for k in new_map if (lid := parse_layer_id(k)) is not None and lid > DST_LAYERS})
    if bad:
        fail(f"重建后的 index 仍包含越界层: {bad[:10]}（> DST_LAYERS={DST_LAYERS}）")
    if has_mtp:
        renamed = sum(1 for k in new_map if parse_layer_id(k) == DST_LAYERS)
        if DST_LAYERS < orig_layers and renamed != len(src_mtp_keys):
            fail(
                f"MTP 改名数量不一致: SRC layer {orig_layers} 有 {len(src_mtp_keys)} 项，"
                f"重建后 layer {DST_LAYERS} 只有 {renamed} 项"
            )
        if f"{LAYER_PREFIX}{DST_LAYERS}.eh_proj.weight" not in new_map:
            fail(f"重建后缺少 {LAYER_PREFIX}{DST_LAYERS}.eh_proj.weight，nextn 前缀会匹配不到草稿层")

    # step5: 重新生成 config.json
    for field in CONFIG_LAYER_LISTS:
        if isinstance(config.get(field), list):
            config[field] = config[field][:DST_LAYERS]
    config["num_hidden_layers"] = DST_LAYERS
    with open(os.path.join(dst_abs, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"[step5] config: num_hidden_layers {orig_layers} -> {DST_LAYERS}")

    # step6: 重新生成 quant_model_description.json（仅 w8a8 目录存在；
    #   model_quant_type/version/is_rot_used 等非 layer key 由 classify 原样保留）
    if has_desc:
        with open(os.path.join(src_abs, desc_name), encoding="utf-8") as f:
            desc = json.load(f)
        new_desc = {}
        for k, v in desc.items():
            nk = classify(k, orig_layers, DST_LAYERS)
            if nk is not None:
                new_desc[nk] = v
        with open(os.path.join(dst_abs, desc_name), "w", encoding="utf-8") as f:
            json.dump(new_desc, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"[step6] 量化描述: 剩余 {len(new_desc)} 项")

    # step7: 拷贝其他文件（tokenizer / 量化 yaml / optional/ 等；config/index/desc 已重新生成）
    regen = {"config.json", index_name} | ({desc_name} if has_desc else set())
    for name in sorted(os.listdir(src_abs)):
        s = os.path.join(src_abs, name)
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(dst_abs, name), dirs_exist_ok=True)
        elif os.path.isfile(s) and not name.endswith(".safetensors") and name not in regen and not name.endswith(".bak"):
            shutil.copy2(s, os.path.join(dst_abs, name))

    # step8: 将本脚本自身拷贝到 DST（模式改为 update），供后续修改层数使用
    self_path = os.path.abspath(__file__)
    dst_script = os.path.join(dst_abs, os.path.basename(self_path))
    if os.path.abspath(dst_script) != self_path:
        with open(self_path, encoding="utf-8") as f:
            content = f.read()
        new_content, n = re.subn(r'^SCRIPT_MODE\s*=\s*("init"|"update").*$',
                                 'SCRIPT_MODE = "update"', content, count=1, flags=re.M)
        if n != 1:
            fail("未在脚本配置区找到 SCRIPT_MODE，无法生成 update 副本")
        with open(dst_script, "w", encoding="utf-8") as f:
            f.write(new_content)
        shutil.copymode(self_path, dst_script)
        print(f"[step8] 已拷贝自身到 {dst_script}（SCRIPT_MODE = update）")

    print("裁剪完成！")


def main():
    if SCRIPT_MODE in ("init", "update"):
        # 两种模式行为一致：均从 SRC 全量重建 DST（幂等）
        build_cut()
    else:
        fail(f'未知的 SCRIPT_MODE: {SCRIPT_MODE!r}（应为 "init" 或 "update"）')


if __name__ == "__main__":
    main()