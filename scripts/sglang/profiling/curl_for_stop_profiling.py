#!/usr/bin/env python3
""":"
echo "错误：请使用 python 而不是 bash 运行此脚本！" >&2; exit 1
"""

import requests

SERVER = "http://127.0.0.1:8880"

url = f"{SERVER}/stop_profile"

try:
    response = requests.post(url)
    response.raise_for_status()

    print("Stop profiling finish")
    print("Response:", response.text)

except requests.exceptions.RequestException as e:
    print("Request failed:", e)
