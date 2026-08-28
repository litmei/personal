""":"
echo "错误：请使用 python 而不是 bash 运行此脚本！" >&2; exit 1

有什么用？
比对 ModelScope 与 HuggingFace 上同一模型的文件 SHA256 一致性。

如何使用？
1. 按照下方“手动填写区域”的指示完成配置
2. 运行该脚本 python xx.py

依赖：
    pip install requests huggingface_hub
"""

import requests
from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile

# ═══════════════════════════════════════════════════════
# ▼▼▼  手动填写区域  ▼▼▼
# ═══════════════════════════════════════════════════════

model_repo_for_modelscope  = "meituan/DeepSeek-R1-Channel-INT8"
model_repo_for_huggingface = "meituan/DeepSeek-R1-Channel-INT8"

# 只比对特定后缀（设为 None 则比对所有文件）
FILE_SUFFIX_FILTER = [".safetensors", ".json", ".py", ".yaml"]

# ═══════════════════════════════════════════════════════
# ▲▲▲  手动填写区域结束  ▲▲▲
# ═══════════════════════════════════════════════════════


def should_check(filename: str) -> bool:
    if FILE_SUFFIX_FILTER is None:
        return True
    return any(filename.endswith(s) for s in FILE_SUFFIX_FILTER)


# ── ModelScope：直接调 HTTP API ──────────────────────────
def get_modelscope_sha() -> dict[str, str]:
    print(f"[ModelScope] 正在获取文件列表: {model_repo_for_modelscope} ...")

    url = f"https://modelscope.cn/api/v1/models/{model_repo_for_modelscope}/repo/files"
    params = {
        "Revision": "master",
        "Root": "",
        "Recursive": "true",
    }
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    files = resp.json()["Data"]["Files"]

    result = {}
    for f in files:
        if f.get("Type") == "tree":       # 跳过目录
            continue
        name = f["Path"]                   # 用 Path（含子目录路径）
        sha  = f.get("Sha256", "")
        if sha and should_check(name):
            result[name] = sha.lower()

    print(f"[ModelScope] 获取到 {len(result)} 个文件")
    return result


# ── HuggingFace：用官方 SDK ──────────────────────────────
def get_huggingface_sha() -> dict[str, str]:
    print(f"[HuggingFace] 正在获取文件列表: {model_repo_for_huggingface} ...")
    api = HfApi()

    result = {}
    for item in api.list_repo_tree(model_repo_for_huggingface, recursive=True):
        if not isinstance(item, RepoFile):
            continue
        name = item.path
        if not should_check(name):
            continue

        if item.lfs and hasattr(item.lfs, "sha256") and item.lfs.sha256:
            result[name] = item.lfs.sha256.lower()
        else:
            # 非 LFS 小文件只有 git blob sha1，无法比对 sha256
            result[name] = f"(git-sha1){item.blob_id}"

    print(f"[HuggingFace] 获取到 {len(result)} 个文件")
    return result


# ── 比对 ─────────────────────────────────────────────────
def compare(ms_sha: dict[str, str], hf_sha: dict[str, str]):
    all_names = sorted(set(ms_sha) | set(hf_sha))

    ok = mismatch = ms_only = hf_only = skip = 0

    print(f"\n{'═' * 70}")
    print(f"  比对结果  (共 {len(all_names)} 个文件)")
    print(f"{'═' * 70}\n")

    for name in all_names:
        in_ms, in_hf = name in ms_sha, name in hf_sha

        if in_ms and not in_hf:
            print(f"  ⚠️  仅 ModelScope 有:  {name}")
            ms_only += 1
        elif in_hf and not in_ms:
            print(f"  ⚠️  仅 HuggingFace 有: {name}")
            hf_only += 1
        else:
            ms_val, hf_val = ms_sha[name], hf_sha[name]
            if hf_val.startswith("(git-sha1)"):
                print(f"  ⏭️  跳过(非LFS):  {name}")
                skip += 1
            elif ms_val == hf_val:
                print(f"  ✅ OK:       {name}")
                ok += 1
            else:
                print(f"  ❌ MISMATCH: {name}")
                print(f"       MS : {ms_val}")
                print(f"       HF : {hf_val}")
                mismatch += 1

    print(f"\n{'─' * 70}")
    print(f"  通过: {ok}  |  不匹配: {mismatch}  |  "
          f"仅MS: {ms_only}  |  仅HF: {hf_only}  |  跳过: {skip}")
    print(f"{'─' * 70}")

    if mismatch == 0 and ms_only == 0 and hf_only == 0:
        print("\n🎉 全部一致！")
    else:
        print("\n⚠️  存在差异，请检查上述输出。")


if __name__ == "__main__":
    ms_sha = get_modelscope_sha()
    hf_sha = get_huggingface_sha()
    compare(ms_sha, hf_sha)