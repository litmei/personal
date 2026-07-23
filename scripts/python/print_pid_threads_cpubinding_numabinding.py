#!/usr/bin/env python3
"""
print_pid_threads_cpubinding_numabinding.py

根据传入的 PID，获取所有线程的 TID，打印 CPU 亲和性和 NUMA 内存绑定信息。

Usage:
    python print_pid_threads_cpubinding_numabinding.py <PID>
    python print_pid_threads_cpubinding_numabinding.py <PID> --watch 2   # 每2秒刷新

Example:
    python print_pid_threads_cpubinding_numabinding.py 12345
"""

import sys
import os
import argparse
import time
import ctypes
import ctypes.util
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────
# 数据采集
# ─────────────────────────────────────────────────────────────────────

def get_all_tids(pid: int) -> list[int]:
    """从 /proc/<pid>/task/ 获取所有线程 TID"""
    task_dir = Path(f"/proc/{pid}/task")
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Process {pid} not found (no /proc/{pid}/task)")
    return sorted(int(d.name) for d in task_dir.iterdir() if d.name.isdigit())


def get_process_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return raw.replace(b"\x00", b" ").decode(errors="replace").strip()
    except Exception:
        return "<unreadable>"


def get_process_name(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except Exception:
        return "?"


def _parse_status_field(pid: int, tid: int, field: str) -> str:
    """从 /proc/<pid>/task/<tid>/status 中解析指定字段"""
    try:
        with open(f"/proc/{pid}/task/{tid}/status") as f:
            for line in f:
                if line.startswith(field):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "?"


def get_thread_name(pid: int, tid: int) -> str:
    try:
        return Path(f"/proc/{pid}/task/{tid}/comm").read_text().strip()
    except Exception:
        return "?"


def get_thread_state(pid: int, tid: int) -> str:
    return _parse_status_field(pid, tid, "State:")


def get_cpu_affinity_list(pid: int, tid: int) -> str:
    """Cpus_allowed_list，如 '0-15' 或 '0,2,4-7'"""
    return _parse_status_field(pid, tid, "Cpus_allowed_list:")


def get_cpu_affinity_mask(pid: int, tid: int) -> str:
    """Cpus_allowed（hex bitmask）"""
    return _parse_status_field(pid, tid, "Cpus_allowed:")


def get_mems_allowed_list(pid: int, tid: int) -> str:
    """Mems_allowed_list，如 '0' 或 '0-1'"""
    return _parse_status_field(pid, tid, "Mems_allowed_list:")


def get_mems_allowed_mask(pid: int, tid: int) -> str:
    """Mems_allowed（hex bitmask）"""
    return _parse_status_field(pid, tid, "Mems_allowed:")


def get_current_cpu(pid: int, tid: int) -> str:
    """线程当前运行在哪个 CPU 核心上（从 /proc/<pid>/task/<tid>/stat 解析）"""
    try:
        content = Path(f"/proc/{pid}/task/{tid}/stat").read_text()
        # comm 字段可能含空格和括号，找最后一个 ')' 跳过
        idx = content.rfind(")")
        fields = content[idx + 2:].split()
        # fields[0]=state(3), ..., fields[36]=processor(39)
        # 1-indexed field 39 → 从 state 起第 37 个 → index 36
        return fields[36]
    except Exception:
        return "?"


def get_cpu_affinity_via_sched(tid: int) -> str:
    """尝试用 os.sched_getaffinity 获取（需要权限）"""
    try:
        cpus = os.sched_getaffinity(tid)
        return format_cpu_set(cpus)
    except (PermissionError, ProcessLookupError, OSError):
        return None


def format_cpu_set(cpus: set[int]) -> str:
    """将 {0,1,2,3,8,9} 格式化为 '0-3,8-9'"""
    if not cpus:
        return "<empty>"
    sorted_cpus = sorted(cpus)
    ranges = []
    start = prev = sorted_cpus[0]
    for c in sorted_cpus[1:]:
        if c == prev + 1:
            prev = c
        else:
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = c
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


# ─────────────────────────────────────────────────────────────────────
# NUMA 拓扑
# ─────────────────────────────────────────────────────────────────────

def get_numa_topology() -> dict[int, str]:
    """返回 {node_id: cpulist_str}"""
    topology = {}
    node_dir = Path("/sys/devices/system/node")
    if not node_dir.is_dir():
        return topology
    for entry in sorted(node_dir.iterdir()):
        if entry.name.startswith("node") and entry.name[4:].isdigit():
            node_id = int(entry.name[4:])
            cpulist_file = entry / "cpulist"
            if cpulist_file.exists():
                topology[node_id] = cpulist_file.read_text().strip()
    return topology


def get_numa_meminfo() -> dict[int, dict[str, int]]:
    """从 /sys/devices/system/node/nodeX/meminfo 获取每个 node 的内存信息 (MB)"""
    info = {}
    node_dir = Path("/sys/devices/system/node")
    if not node_dir.is_dir():
        return info
    for entry in sorted(node_dir.iterdir()):
        if entry.name.startswith("node") and entry.name[4:].isdigit():
            node_id = int(entry.name[4:])
            meminfo_file = entry / "meminfo"
            if meminfo_file.exists():
                node_info = {}
                for line in meminfo_file.read_text().splitlines():
                    parts = line.split()
                    # 格式: "Node 0 MemTotal:       64000 MB"
                    if len(parts) >= 4:
                        key = parts[2].rstrip(":")
                        val = int(parts[3])
                        node_info[key] = val
                info[node_id] = node_info
    return info


# ─────────────────────────────────────────────────────────────────────
# cgroup 信息（可选）
# ─────────────────────────────────────────────────────────────────────

def get_cgroup_cpuset(pid: int, tid: int) -> str:
    """尝试读取 cgroup v2 的 cpuset"""
    try:
        # cgroup v2
        cgroup_file = Path(f"/proc/{pid}/task/{tid}/cgroup")
        for line in cgroup_file.read_text().splitlines():
            # 格式: "0::/path"
            parts = line.split(":", 2)
            if len(parts) == 3 and parts[0] == "0":
                cgroup_path = parts[2]
                cpuset_file = Path(f"/sys/fs/cgroup{cgroup_path}/cpuset.cpus.effective")
                if cpuset_file.exists():
                    return cpuset_file.read_text().strip()
    except Exception:
        pass
    return ""


# ─────────────────────────────────────────────────────────────────────
# 打印
# ─────────────────────────────────────────────────────────────────────

def print_report(pid: int, verbose: bool = False):
    tids = get_all_tids(pid)
    cmdline = get_process_cmdline(pid)
    pname = get_process_name(pid)
    numa_topo = get_numa_topology()
    numa_mem = get_numa_meminfo()

    # ── 进程概览 ──
    print("=" * 100)
    print(f"  PID      : {pid}")
    print(f"  Name     : {pname}")
    print(f"  Cmdline  : {cmdline}")
    print(f"  Threads  : {len(tids)}")
    print("=" * 100)

    # ── NUMA 拓扑 ──
    if numa_topo:
        print("\n  NUMA Topology:")
        for node_id, cpus in numa_topo.items():
            mem_str = ""
            if node_id in numa_mem:
                total = numa_mem[node_id].get("MemTotal", 0)
                free = numa_mem[node_id].get("MemFree", 0)
                mem_str = f"  (Mem: {total} MB total, {free} MB free)"
            print(f"    Node {node_id}: CPUs [{cpus}]{mem_str}")
    else:
        print("\n  NUMA: not available or single-node system")

    # ── 进程级亲和性（主线程 = PID）──
    proc_cpu = get_cpu_affinity_list(pid, pid)
    proc_mem = get_mems_allowed_list(pid, pid)
    print(f"\n  Process-level (main thread):")
    print(f"    CPU affinity : {proc_cpu}")
    print(f"    NUMA mems    : {proc_mem}")

    # cgroup
    cg = get_cgroup_cpuset(pid, pid)
    if cg:
        print(f"    cgroup cpuset: {cg}")

    # ── 线程详情表 ──
    print(f"\n  {'─' * 96}")

    # 表头
    hdr = (
        f"  {'TID':<8}"
        f"{'Name':<22}"
        f"{'State':<16}"
        f"{'CurCPU':<7}"
        f"{'CPU Affinity':<22}"
        f"{'NUMA Mems':<14}"
    )
    if verbose:
        hdr += f"{'CPU Mask(hex)':<18}{'Mems Mask(hex)':<16}"
    print(hdr)
    print(f"  {'─' * 96}")

    for tid in tids:
        name = get_thread_name(pid, tid)
        state = get_thread_state(pid, tid)
        cur_cpu = get_current_cpu(pid, tid)
        cpu_aff = get_cpu_affinity_list(pid, tid)
        mem_aff = get_mems_allowed_list(pid, tid)

        # 截断过长的线程名
        if len(name) > 20:
            name = name[:17] + "..."

        row = (
            f"  {tid:<8}"
            f"{name:<22}"
            f"{state:<16}"
            f"{cur_cpu:<7}"
            f"{cpu_aff:<22}"
            f"{mem_aff:<14}"
        )
        if verbose:
            cpu_mask = get_cpu_affinity_mask(pid, tid)
            mem_mask = get_mems_allowed_mask(pid, tid)
            row += f"{cpu_mask:<18}{mem_mask:<16}"

        print(row)

    print(f"  {'─' * 96}")

    # ── 交叉验证（sched_getaffinity）──
    print(f"\n  Cross-check via os.sched_getaffinity():")
    for tid in tids[:5]:  # 只验证前 5 个，避免刷屏
        result = get_cpu_affinity_via_sched(tid)
        name = get_thread_name(pid, tid)
        if result is not None:
            print(f"    TID {tid} ({name}): {result}")
        else:
            print(f"    TID {tid} ({name}): <no permission>")
    if len(tids) > 5:
        print(f"    ... and {len(tids) - 5} more threads")

    print()


# ─────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Print CPU affinity and NUMA binding for all threads of a process."
    )
    parser.add_argument("pid", type=int, help="Target process ID")
    parser.add_argument(
        "--watch", "-w", type=float, default=0, metavar="SEC",
        help="Refresh every SEC seconds (like top). 0 = print once."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Also show hex bitmasks for CPU and NUMA masks."
    )
    args = parser.parse_args()

    pid = args.pid

    # 检查进程是否存在
    if not Path(f"/proc/{pid}").is_dir():
        print(f"Error: Process {pid} does not exist.", file=sys.stderr)
        sys.exit(1)

    if args.watch > 0:
        try:
            while True:
                os.system("clear" if os.name != "nt" else "cls")
                print_report(pid, verbose=args.verbose)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        print_report(pid, verbose=args.verbose)


if __name__ == "__main__":
    main()