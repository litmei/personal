#!/usr/bin/env python3
""":"
echo "错误：请使用 python 而不是 bash 运行此脚本！" >&2; exit 1
"""

import argparse
from datetime import datetime

import requests


SERVER = "http://127.0.0.1:8880"
BASE_DIR = "/mnt/share/xjw/prof"


def start_profiling():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    profile_dir = f"{BASE_DIR}/prof_{timestamp}"

    payload = {
        "num_steps": 5,
        "output_dir": profile_dir,
        "activities": ["CPU", "GPU"],
        "record_shapes": False,
        "profile_memory": False,
        "with_stack": False,
        "profile_prefix": "prof-",
    }

    url = f"{SERVER}/start_profile"
    headers = {"Content-Type": "application/json"}

    print(f"Starting profiling...")
    print(f"Output directory: {profile_dir}")

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()

        print("Start profiling finish")
        print("Response:", response.text)

    except requests.exceptions.RequestException as e:
        print("Request failed:", e)


def stop_profiling():
    url = f"{SERVER}/stop_profile"

    print("Stopping profiling...")

    try:
        response = requests.post(url)
        response.raise_for_status()

        print("Stop profiling finish")
        print("Response:", response.text)

    except requests.exceptions.RequestException as e:
        print("Request failed:", e)


def main():
    parser = argparse.ArgumentParser(
        description="SGLang profiling control script"
    )

    parser.add_argument(
        "action",
        nargs="?",
        choices=["start", "stop"],
        default="start",
        help="profiling action: start or stop (default: start)",
    )

    args = parser.parse_args()

    if args.action == "start":
        start_profiling()
    elif args.action == "stop":
        stop_profiling()


if __name__ == "__main__":
    main()
