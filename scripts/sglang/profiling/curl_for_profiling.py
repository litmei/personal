#!/usr/bin/env python3
import requests
from datetime import datetime

SERVER = "http://0.0.0.0:6677"
BASE_DIR = "/mnt/share/xjw/prof"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
PROFILE_DIR = f"{BASE_DIR}/prof_{TIMESTAMP}"

payload = {
    "num_steps": 5,
    "output_dir": PROFILE_DIR,
    "activities": ["CPU", "GPU"],
    "record_shapes": True,
    "profile_memory": True,
    "with_stack": False,
    "profile_prefix": "glm52-",
}

url = f"{SERVER}/start_profile"
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    print("Prof finish")
except requests.exceptions.RequestException as e:
    print("Request failed:", e)
