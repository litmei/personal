#!/usr/bin/env python3
""":"
echo "错误：请使用 python 而不是 bash 运行此脚本！" >&2; exit 1
"""

from torch_npu.profiler.profiler import analyse

path = "./result_data"

analyse(profiler_path=path)
