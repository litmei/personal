#!/usr/bin/env python3

# python print_pid_threads_with_gdb.py --pid <pid> --packages <mooncake [zmq]>

import os
import sys
import re
import argparse
import subprocess
import importlib


def get_package_paths(package_names):
    """
    尝试导入包并获取其 __path__ 或文件所在目录
    """
    paths = []
    # 将当前路径加入 sys.path，以便能导入当前目录下的本地包
    if "" not in sys.path:
        sys.path.insert(0, "")

    for name in package_names:
        try:
            pkg = importlib.import_module(name)
            pkg_paths = []
            if hasattr(pkg, "__path__"):
                # 如果是包，获取 __path__
                pkg_paths = list(pkg.__path__)
            elif hasattr(pkg, "__file__") and pkg.__file__:
                # 如果是单文件模块，获取其所在的目录
                pkg_paths = [os.path.dirname(os.path.abspath(pkg.__file__))]

            # 标准化路径
            normalized_paths = [os.path.realpath(os.path.abspath(p)) for p in pkg_paths]
            print(f"[+] 成功导入包 '{name}'，解析到路径: {normalized_paths}")
            paths.extend(normalized_paths)
        except ImportError:
            print(f"[-] 警告: 无法导入包 '{name}'，将跳过此包的路径解析。")

    # 去重
    return list(set(paths))


def run_gdb_and_get_threads(pid, solib_paths):
    """
    配置 solib-search-path，attach 到进程并执行 info threads
    """
    solib_search_path = ":".join(solib_paths)

    # 构造 GDB 命令
    # -batch 模式可以让 GDB 执行完命令后自动退出（自动 detach）
    gdb_cmd = [
        "gdb",
        "-batch",
        "-ex", f"set solib-search-path {solib_search_path}",
        "-ex", f"attach {pid}",
        "-ex", "info threads"
    ]

    print(f"[+] 正在调用 GDB 挂载 PID {pid} 并检索线程信息...")
    try:
        result = subprocess.run(
            gdb_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60
        )
        if result.returncode != 0:
            print(f"[-] GDB 执行失败 (错误码 {result.returncode}):")
            print(result.stderr)
            return ""
        return result.stdout
    except subprocess.TimeoutExpired:
        print("[-] 错误: GDB 执行超时。")
        return ""
    except FileNotFoundError:
        print("[-] 错误: 系统中未找到 'gdb' 命令，请确保已安装 GDB。")
        return ""


def parse_target_threads(gdb_output, target_paths):
    """
    解析 GDB 输出，提取包含指定包路径的线程 ID (LWP)
    """
    # 匹配 info threads 中的 LWP (Thread ID)
    # 典型格式如: * 1    Thread 0x7f9a1b2c3d40 (LWP 12345) "main" ...
    lwp_pattern = re.compile(r'LWP\s+(\d+)')
    matched_threads = []

    for line in gdb_output.splitlines():
        match = lwp_pattern.search(line)
        if match:
            tid = int(match.group(1))
            # 检查当前行是否包含任何一个目标包路径
            is_matched = False
            for path in target_paths:
                if path in line:
                    is_matched = True
                    break
            if is_matched:
                matched_threads.append((tid, line.strip()))

    return matched_threads


def get_thread_details(pid, tid):
    """
    读取 /proc/<pid>/task/<tid>/status 获取线程名称、绑核和 NUMA 亲和性信息
    """
    status_path = f"/proc/{pid}/task/{tid}/status"
    name = "未知"
    cpus_allowed = "未知"
    mems_allowed = "未知"

    if not os.path.exists(status_path):
        return "线程已退出", "线程可能已退出", "线程可能已退出"

    try:
        with open(status_path, "r") as f:
            for line in f:
                if line.startswith("Name:"):
                    name = line.split(":", 1)[1].strip()
                elif line.startswith("Cpus_allowed_list:"):
                    cpus_allowed = line.split(":", 1)[1].strip()
                elif line.startswith("Mems_allowed_list:"):
                    mems_allowed = line.split(":", 1)[1].strip()
    except Exception as e:
        return "读取失败", f"读取失败 ({str(e)})", f"读取失败 ({str(e)})"

    return name, cpus_allowed, mems_allowed


def main():
    parser = argparse.ArgumentParser(description="分析特定 Python 包线程的绑核和 NUMA 亲和信息")
    parser.add_argument("--pid", type=int, required=True, help="目标进程的 PID")
    parser.add_argument("--packages", nargs="+", required=True, help="待检测的 Python 包名，支持传入多个（空格分隔）")

    args = parser.parse_args()

    # 1. 检查 Linux 系统
    if not sys.platform.startswith("linux"):
        print("[-] 错误: 本脚本仅支持 Linux 系统。")
        sys.exit(1)

    # 2. 解析包路径
    target_paths = get_package_paths(args.packages)
    if not target_paths:
        print("[-] 未能解析到任何有效的包路径，脚本退出。")
        sys.exit(1)

    # 3. 运行 GDB 并获取输出
    gdb_output = run_gdb_and_get_threads(args.pid, target_paths)
    if not gdb_output:
        sys.exit(1)

    # 4. 解析目标线程
    matched_threads = parse_target_threads(gdb_output, target_paths)
    if not matched_threads:
        print("[*] 未在 GDB 线程列表中找到匹配指定包路径的线程。")
        sys.exit(0)

    # 5. 获取并打印亲和性信息
    print(f"\n[+] 发现以下 {len(matched_threads)} 个相关线程:")
    print("-" * 115)
    print(
        f"{'Thread Name':<20} | {'Thread ID (LWP)':<16} | {'CPU Affinity (Cpus_allowed_list)':<32} | {'NUMA Affinity (Mems_allowed_list)':<32}")
    print("-" * 115)

    for tid, gdb_line in matched_threads:
        name, cpu_aff, numa_aff = get_thread_details(args.pid, tid)
        print(f"{name:<20} | {tid:<16} | {cpu_aff:<32} | {numa_aff:<32}")

    print("-" * 115)


if __name__ == "__main__":
    main()


