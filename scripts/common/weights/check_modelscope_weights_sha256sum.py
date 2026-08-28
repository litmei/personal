""":"
echo "错误：请使用 python 而不是 bash 运行此脚本！" >&2; exit 1

有什么用？
比对本地模型文件与 ModelScope 远程仓库的 SHA256 一致性。
逐条计算、逐条比对、逐条打印。

如何使用？
1. 按照下方“手动填写区域”的指示完成配置
2. 运行该脚本 python xx.py

依赖：
    pip install requests
"""

import os
import subprocess
import requests

# ═══════════════════════════════════════════════════════
# ▼▼▼  手动填写区域  ▼▼▼
# ═══════════════════════════════════════════════════════

# 填写例如 "Eco-Tech/DeepSeek-V3.2-Exp-w4a8-mtp-QuaRot" 的字符串repo
model_repo_for_modelscope = None

# 本地模型文件所在目录。
#   - 填路径（如 "/data/models/DeepSeek-V3.2-Exp-w4a8-mtp-QuaRot"）→ 自动执行 sha256sum
#     默认值为 "./" 方便快速拷贝本文件后使用
#   - 填 None → 使用下方 local_sha256_name_lines 手动粘贴的结果
local_path = "./"

# 当 local_path = None 时，将 `sha256sum ./*.safetensors` 的输出粘贴到此处以生效，示例如下
local_sha256_name_lines = r"""
2703effcc3d695b64ca807ba6d6d3a4e605631713c825d2c60e44717041bb983  ./quant_model_weights-00083-of-00088.safetensors
fcc74f5b5f4addc6720a390a5549ae0e7d18a500642b56a58268a33a93ea32fc  ./quant_model_weights-00084-of-00088.safetensors
71a79c819aef24ad2e1238c3c82e16a414336f1d55f5f5949df2b54318661ed7  ./quant_model_weights-00085-of-00088.safetensors
e6fd2d7dc113bf3408c1b270953d2f56109b9d20e6f10abef2dd8801de8cf40a  ./quant_model_weights-00086-of-00088.safetensors
"""

# 只比对特定后缀（设为 None 则比对所有文件）
FILE_SUFFIX_FILTER = [".safetensors", ".json", ".txt", ".model"]

# ═══════════════════════════════════════════════════════
# ▲▲▲  手动填写区域结束  ▲▲▲
# ═══════════════════════════════════════════════════════

LONGEST_NAME_LEN = 0

assert type(model_repo_for_modelscope) == str, "请手动补充模型repo在model_repo_for_modelscope顶参中"


def should_check(filename: str) -> bool:
    if FILE_SUFFIX_FILTER is None:
        return True
    return any(filename.endswith(s) for s in FILE_SUFFIX_FILTER)


# ── 获取 ModelScope 远程 SHA256 ──────────────────────────
def get_modelscope_sha() -> dict[str, str]:
    global LONGEST_NAME_LEN

    print(f"[ModelScope] 正在获取文件列表: {model_repo_for_modelscope} ...")

    url = f"https://modelscope.cn/api/v1/models/{model_repo_for_modelscope}/repo/files"
    params = {"Revision": "master", "Root": "", "Recursive": "true"}
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    files = resp.json()["Data"]["Files"]

    result = {}
    for f in files:
        if f.get("Type") == "tree":
            continue
        name = f["Path"]
        sha = f.get("Sha256", "")
        if sha and should_check(name):
            result[name] = sha.lower()
            LONGEST_NAME_LEN = max(LONGEST_NAME_LEN, len(name))

    print(f"[ModelScope] 获取到 {len(result)} 个文件\n")
    return result


# ── 单条比对并打印 ───────────────────────────────────────
def compare_one(name: str, local_sha: str, remote_sha: dict[str, str]) -> str:
    """比对单个文件，打印结果，返回状态: 'ok' / 'mismatch' / 'local_only'"""
    if name not in remote_sha:
        print(f"  ⚠️  仅本地有（远程缺失）: {name}")
        print(f"       Local : {local_sha}")
        return "local_only"

    remote_val = remote_sha[name]
    if local_sha == remote_val:
        print(f"  ✅ OK: {name:<{LONGEST_NAME_LEN}}  {local_sha}")
        return "ok"
    else:
        print(f"  ❌ MISMATCH: {name}")
        print(f"       Local : {local_sha}")
        print(f"       Remote: {remote_val}")
        return "mismatch"


# ── 模式 A：从粘贴文本逐行比对 ───────────────────────────
def run_from_lines(lines: str, remote_sha: dict[str, str]):
    parsed_lines = [l.strip() for l in lines.strip().splitlines() if l.strip()]
    if not parsed_lines:
        raise ValueError(
            "local_path 为 None 时，local_sha256_name_lines 不能为空！\n"
            "请先在机器上执行 sha256sum 并将结果粘贴到脚本中。"
        )

    total = len(parsed_lines)
    stats = {"ok": 0, "mismatch": 0, "local_only": 0}
    checked_names = set()

    print(f"[Local] 从粘贴文本解析到 {total} 条记录，开始逐条比对...\n")

    for idx, line in enumerate(parsed_lines, 1):
        parts = line.split()
        sha = parts[0].lower()
        name = parts[1].lstrip("./")
        if not should_check(name):
            continue

        print(f"[{idx:^4}/{total:^4}]", end="")
        status = compare_one(name, sha, remote_sha)
        stats[status] += 1
        checked_names.add(name)

    # 检查远程有而本地没有的
    remote_only = set(remote_sha.keys()) - checked_names
    for name in sorted(remote_only):
        print(f"  ⚠️  仅远程有（本地缺失）: {name}")
        print(f"       Remote: {remote_sha[name]}")

    return stats, len(remote_only)


# ── 模式 B：从目录逐文件计算并比对 ───────────────────────
def run_from_path(path: str, remote_sha: dict[str, str]):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"目录不存在: {path}")

    filenames = sorted(
        f for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) and should_check(f)
    )
    if not filenames:
        raise FileNotFoundError(f"目录中没有匹配的文件: {path}")

    total = len(filenames)
    stats = {"ok": 0, "mismatch": 0, "local_only": 0}
    checked_names = set()

    print(f"[Local] 目录 {path} 下共 {total} 个文件，开始逐个计算 SHA256 并比对...\n")

    for idx, fname in enumerate(filenames, 1):
        filepath = os.path.join(path, fname)
        print(f"[{idx:^4}/{total:^4}] 计算中: {fname} ...", end="", flush=True)

        # 计算 sha256
        proc = subprocess.run(
            ["sha256sum", filepath],
            capture_output=True, text=True, check=True,
        )
        sha = proc.stdout.split()[0].lower()

        # 覆盖掉 "计算中..." 那一行，打印比对结果
        print(f"\r[{idx:^4}/{total:^4}]", end="")
        status = compare_one(fname, sha, remote_sha)
        stats[status] += 1
        checked_names.add(fname)

    # 检查远程有而本地没有的
    remote_only = set(remote_sha.keys()) - checked_names
    for name in sorted(remote_only):
        print(f"  ⚠️  仅远程有（本地缺失）: {name}")
        print(f"       Remote: {remote_sha[name]}")

    return stats, len(remote_only)


# ── 汇总 ─────────────────────────────────────────────────
def print_summary(stats: dict, remote_only_count: int):
    total_checked = stats["ok"] + stats["mismatch"] + stats["local_only"]
    print(f"\n{'─' * 70}")
    print(f"  通过: {stats['ok']}  |  不匹配: {stats['mismatch']}  |  "
          f"仅本地: {stats['local_only']}  |  仅远程: {remote_only_count}")
    print(f"{'─' * 70}")

    if stats["mismatch"] == 0 and stats["local_only"] == 0 and remote_only_count == 0:
        print("\n🎉 全部一致！")
    else:
        print("\n⚠️  存在差异，请检查上述输出。")


# ── 主流程 ───────────────────────────────────────────────
if __name__ == "__main__":
    remote_sha = get_modelscope_sha()

    if local_path is not None:
        stats, remote_only_count = run_from_path(local_path, remote_sha)
    else:
        stats, remote_only_count = run_from_lines(local_sha256_name_lines, remote_sha)

    print_summary(stats, remote_only_count)