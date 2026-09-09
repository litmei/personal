import argparse
import re
from typing import Dict, List

import matplotlib.pyplot as plt
from safetensors import safe_open

# 文件命名格式: norm_input_r0_{global_id}_layer{layer_id}_{norm_name}.safetensors
FILE_PATTERN = re.compile(
    r"^norm_input_r0_(\d+)_layer(\d+)_(.+)\.safetensors$"
)


def parse_weight_files(weight_dir: str, filter_name: str = None) -> List[Dict]:
    """扫描目录, 解析文件名并按 global_id 排序返回文件信息列表"""
    import os

    entries = []
    for fname in os.listdir(weight_dir):
        match = FILE_PATTERN.match(fname)
        if not match:
            continue
        global_id = int(match.group(1))
        layer_id = int(match.group(2))
        norm_name = match.group(3)
        if filter_name and norm_name != filter_name:
            continue
        entries.append(
            {
                "path": os.path.join(weight_dir, fname),
                "global_id": global_id,
                "layer_id": layer_id,
                "norm_name": norm_name,
            }
        )
    entries.sort(key=lambda x: x["global_id"])
    return entries


def collect_residual_stats(entries: List[Dict]) -> List[Dict]:
    """逐个加载权重文件, 统计 tensor['residual'] 的 max/min/mean"""
    stats = []
    for entry in entries:
        with safe_open(entry["path"], framework="numpy") as f:
            residual = f.get_tensor("residual")
            stats.append(
                {
                    "global_id": entry["global_id"],
                    "layer_id": entry["layer_id"],
                    "norm_name": entry["norm_name"],
                    "max": float(residual.max()),
                    "min": float(residual.min()),
                    "mean": float(residual.mean()),
                }
            )
        print(
            f"[global_id={entry['global_id']}] layer{entry['layer_id']} "
            f"{entry['norm_name']}: max={stats[-1]['max']:.6f}, "
            f"min={stats[-1]['min']:.6f}, mean={stats[-1]['mean']:.6f}"
        )
    return stats


def plot_stats(stats: List[Dict], save_path: str = None):
    """绘制 residual 的 max/min/mean 曲线, x 轴为 global_id"""
    global_ids = [s["global_id"] for s in stats]
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    items = [("max", "tab:red"), ("min", "tab:blue"), ("mean", "tab:green")]
    for ax, (key, color) in zip(axes, items):
        ax.plot(global_ids, [s[key] for s in stats], marker="o", color=color)
        ax.set_ylabel(key)
        ax.grid(True)

    norm_names = sorted({s["norm_name"] for s in stats})
    title = f"residual stats ({', '.join(norm_names)})"
    axes[0].set_title(title)
    axes[-1].set_xlabel("global_id")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="加载 norm dump 权重并绘制 residual 的 max/min/mean 统计图"
    )
    parser.add_argument(
        "--dir", required=True, help="权重文件所在目录"
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="按 norm_name 过滤, 例如 input_layernorm, 不指定则加载全部",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="图表保存路径 (如 stats.png), 不指定则弹窗展示",
    )
    args = parser.parse_args()

    entries = parse_weight_files(args.dir, args.filter)
    if not entries:
        print("未找到匹配的权重文件")
        return
    print(f"共找到 {len(entries)} 个权重文件")

    stats = collect_residual_stats(entries)
    plot_stats(stats, args.save)


if __name__ == "__main__":
    main()
