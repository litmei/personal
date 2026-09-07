""":"
echo "错误：请使用 python 而不是 bash 运行此脚本！" >&2; exit 1

有什么用？
对GLM5.2模型进行裁剪

如何使用？
1. 按照下方“手动填写区域”的指示完成配置
2. 运行该脚本 python xx.py
3. SCRIPT_MODE 两种模式：
   - "init"  首次使用：软链接/拷贝 SRC 权重到 DST，并将本脚本自身拷贝到 DST
             （副本的模式会被改为 "update"），随后代跑一次 update 逻辑
   - "update" 修改层数：根据 DST_LAYERS 修改 DST_MODEL_PATH 下的
             config.json 与 model.safetensors.index.json（可反复运行，幂等）
             用法：在 DST 目录下的脚本副本中改好 DST_LAYERS 后运行

依赖：
无（仅 Python 标准库）
"""

# ==================== 手动填写区域 ====================
SRC_MODEL_PATH = r"/home/litmei/workspace/weights/demo/src"
DST_MODEL_PATH = r"/home/litmei/workspace/weights/demo/dst"
DST_LAYERS = 10
SCRIPT_MODE = "init"  # "update"
# =====================================================

import json
import os
import re
import shutil
import sys

LAYER_PREFIX = "model.layers."


def fail(msg):
    print(f"[错误] {msg}")
    sys.exit(1)


def parse_layer_id(key):
    # "model.layers.10.mlp.xxx" -> 10，非 layer 权重返回 None
    if not key.startswith(LAYER_PREFIX):
        return None
    seg = key[len(LAYER_PREFIX):].split(".", 1)[0]
    return int(seg) if seg.isdigit() else None


def restore_or_backup(path):
    # step4.1: 有 .bak 则用 .bak 覆盖当前文件（还原到原始版本）；没有则把当前文件备份为 .bak
    bak = path + ".bak"
    if os.path.isfile(bak):
        shutil.copy2(bak, path)
    else:
        if not os.path.isfile(path):
            fail(f"缺少文件: {path}")
        shutil.copy2(path, bak)


def update_num_layers(model_dir, dst_layers):
    """update 模式核心逻辑（step4.1 ~ step4.3），修改 model_dir 下的裁剪层数"""
    if not isinstance(dst_layers, int) or isinstance(dst_layers, bool) or dst_layers <= 0:
        fail(f"DST_LAYERS 必须是正整数，当前为: {dst_layers!r}")
    config_path = os.path.join(model_dir, "config.json")
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    for path in (config_path, index_path):
        if not os.path.isfile(path):
            fail(f"缺少文件: {path}")

    # step4.1: 还原 / 备份 config.json 与 model.safetensors.index.json
    restore_or_backup(config_path)
    restore_or_backup(index_path)

    # step4.2: 修改 config.json
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    orig_layers = config["num_hidden_layers"]
    if dst_layers > orig_layers:
        fail(f"DST_LAYERS({dst_layers}) 不能大于原始层数 num_hidden_layers({orig_layers})")
    # 所有“每层一项”的列表都要同步截断（mlp_layer_types: dense/sparse, indexer_types: full/shared）
    for field in ("mlp_layer_types", "indexer_types"):
        if isinstance(config.get(field), list):
            config[field] = config[field][:dst_layers]
    config["num_hidden_layers"] = dst_layers
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"[step4.2] num_hidden_layers: {orig_layers} -> {dst_layers}")

    # step4.3: 修改 model.safetensors.index.json
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    new_map = {}
    dropped = renamed = 0
    for key, shard in weight_map.items():
        lid = parse_layer_id(key)
        if lid is None or lid < dst_layers:
            new_map[key] = shard
        elif lid == orig_layers:
            # nextn 层（layer id == 原始 num_hidden_layers）：保留，layer id 改为 dst_layers
            new_map[LAYER_PREFIX + str(dst_layers) + key[len(LAYER_PREFIX) + len(str(lid)):]] = shard
            renamed += 1
        else:
            dropped += 1
    index["weight_map"] = new_map
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    if renamed:
        print(f"[step4.3] 删除被裁剪层权重 {dropped} 项，nextn 层 {renamed} 项重命名为 layer {dst_layers}，共剩余 {len(new_map)} 项")
    else:
        print(f"[提示] 未发现 nextn 层（layer {orig_layers}），仅删除被裁剪层权重 {dropped} 项，共剩余 {len(new_map)} 项")


def run_init():
    # step0: 检查SRC_MODEL_PATH 是否存在
    if not os.path.isdir(SRC_MODEL_PATH):
        fail(f"SRC_MODEL_PATH 不存在: {SRC_MODEL_PATH}")

    # step1: mkdir -p DST_MODEL_PATH
    os.makedirs(DST_MODEL_PATH, exist_ok=True)
    print(f"[step1] 输出目录: {DST_MODEL_PATH}")

    names = sorted(os.listdir(SRC_MODEL_PATH))

    # step2: ln -s SRC_MODEL_PATH/*.safetensors DST_MODEL_PATH/ ，以软链接的方式创建裁剪权重
    for name in names:
        src_file = os.path.join(SRC_MODEL_PATH, name)
        if not name.endswith(".safetensors") or not os.path.isfile(src_file):
            continue
        dst_file = os.path.join(DST_MODEL_PATH, name)
        if os.path.lexists(dst_file):
            os.remove(dst_file)  # 重复运行时先清掉旧的链接/文件
        os.symlink(src_file, dst_file)
        print(f"[step2] 软链接 {name} -> {src_file}")

    # step3: cp SRC_MODEL_PATH/除了safetensors的其他文件 DST_MODEL_PATH/
    for name in names:
        src_file = os.path.join(SRC_MODEL_PATH, name)
        if name.endswith(".safetensors") or not os.path.isfile(src_file):
            continue
        shutil.copy2(src_file, os.path.join(DST_MODEL_PATH, name))
        print(f"[step3] 拷贝 {name}")

    # step4: 将本脚本自身拷贝到 DST_MODEL_PATH/，副本的模式改为 "update"，
    #        供用户之后在 DST 目录中自行改 DST_LAYERS 后运行
    self_path = os.path.abspath(__file__)
    with open(self_path, encoding="utf-8") as f:
        self_content = f.read()
    new_content, n = re.subn(r'^SCRIPT_MODE\s*=\s*"init".*$', 'SCRIPT_MODE = "update"',
                             self_content, count=1, flags=re.M)
    if n != 1:
        fail('未在脚本配置区找到 SCRIPT_MODE = "init"，无法生成 update 副本')
    dst_script = os.path.join(DST_MODEL_PATH, os.path.basename(self_path))
    with open(dst_script, "w", encoding="utf-8") as f:
        f.write(new_content)
    shutil.copymode(self_path, dst_script)
    print(f"[step4] 已拷贝自身到 {dst_script}（SCRIPT_MODE = update）")

    # 代跑一次 update 逻辑（step4.1 ~ step4.3）
    update_num_layers(DST_MODEL_PATH, DST_LAYERS)

    print("裁剪完成！")


def run_update():
    # update 模式：根据 DST_LAYERS 修改 DST_MODEL_PATH 路径下的内容
    if not os.path.isdir(DST_MODEL_PATH):
        fail(f"DST_MODEL_PATH 不存在: {DST_MODEL_PATH}")
    update_num_layers(DST_MODEL_PATH, DST_LAYERS)
    print("层数修改完成！")


def main():
    if SCRIPT_MODE == "init":
        run_init()
    elif SCRIPT_MODE == "update":
        run_update()
    else:
        fail(f'未知的 SCRIPT_MODE: {SCRIPT_MODE!r}（应为 "init" 或 "update"）')


if __name__ == "__main__":
    main()
