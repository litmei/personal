#!/usr/bin/env python3
"""
进程/线程信息查看工具。

用法:
    python print_pid_info.py <pid> <package_name_0> [package_name_1, ...]

示例:
    python print_pid_info.py 123456 mooncake zmq

功能:
    递归获取 PID 的所有 TID，展示 CPU 亲和、当前运行核、并通过 GDB 栈帧检测所属 package。
"""

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

# =========================================================================
# Debug 开关：开启时打印 GDB 原始输出、解析结果等调试信息
# =========================================================================
DEBUG_GDB = False

# =========================================================================
# 硬编码的符号搜索路径（用户按需填写）
# 这些路径下的 .so 文件会作为 GDB 的符号搜索路径
# =========================================================================
SYMBOL_SEARCH_PATHS: List[str] = [
    # "/usr/local/python3.11.15/lib/python3.11/site-packages/mooncake",
]


def _read_file(path: str) -> Optional[str]:
    try:
        with open(path) as f:
            return f.read().strip()
    except (FileNotFoundError, PermissionError):
        return None


def _read_stat_field(tid: int, field_index: int) -> Optional[str]:
    """读取 /proc/<tid>/stat 的指定字段（从 1 开始计数）。

    特殊处理: comm 字段可能包含空格，格式为 (comm)。"""
    try:
        with open(f"/proc/{tid}/stat") as f:
            content = f.read()
        # 跳过 (comm) 部分
        comm_end = content.rfind(")")
        rest = content[comm_end + 2:]  # 跳过 ") "
        fields = rest.split()
        return fields[field_index - 3]  # 前 3 个字段: pid, (comm), state 已被跳过
    except (FileNotFoundError, PermissionError, IndexError):
        return None


def get_all_tids(pid: int) -> List[int]:
    """获取进程的所有线程 TID。"""
    try:
        return sorted(int(e) for e in os.listdir(f"/proc/{pid}/task") if e.isdigit())
    except (FileNotFoundError, PermissionError):
        return []


def get_child_pids(pid: int) -> List[int]:
    """获取进程的所有子进程 PID。"""
    content = _read_file(f"/proc/{pid}/task/{pid}/children")
    if not content:
        return []
    return sorted(int(x) for x in content.split())


def get_tid_name(tid: int) -> str:
    """获取线程/进程名称。"""
    name = _read_file(f"/proc/{tid}/comm")
    return name or "unknown"


def get_cpu_affinity(tid: int) -> Set[int]:
    """获取 CPU 亲和集。"""
    try:
        return os.sched_getaffinity(tid)
    except (ProcessLookupError, PermissionError):
        line = _read_file(f"/proc/{tid}/status")
        if line:
            for l in line.split("\n"):
                if l.startswith("Cpus_allowed_list:"):
                    val = l.split(":", 1)[1].strip()
                    return _parse_cpu_list(val)
        return set()


def _parse_cpu_list(val: str) -> Set[int]:
    """解析 "0-3,8-11" 格式的 CPU 列表。"""
    result = set()
    if not val:
        return result
    for part in val.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            result.update(range(int(a), int(b) + 1))
        else:
            result.add(int(part))
    return result


def get_current_cpu(tid: int) -> Optional[int]:
    """获取线程当前正在运行的 CPU 核。

    /proc/<tid>/stat 第 39 个字段是 processor (从 1 开始计数)。"""
    val = _read_stat_field(tid, 39)
    return int(val) if val is not None else None


def fmt_affinity(affinity: Set[int]) -> str:
    """格式化 CPU 亲和集为紧凑表示。"""
    if not affinity:
        return "N/A"
    sorted_cpus = sorted(affinity)
    ranges = []
    start = sorted_cpus[0]
    end = sorted_cpus[0]
    for cpu in sorted_cpus[1:]:
        if cpu == end + 1:
            end = cpu
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = end = cpu
    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ",".join(ranges)


def gdb_get_all_stacks(pid: int, timeout: int = 30) -> Dict[int, str]:
    """一次 GDB attach 获取进程下所有线程的用户态调用栈。

    优化: 对每个线程单独 attach gdb 极慢（N 线程 = N 次 gdb）。
    改为 attach 到 PID 一次，用 'thread apply all bt' 获取全部线程栈，
    通过 '(LWP XXXX)' 解析出每个 TID 的栈帧，返回 {tid: stack_text} 映射。"""
    cmd = ["gdb", "-p", str(pid), "-batch", "-q"]
    if SYMBOL_SEARCH_PATHS:
        search_path = ":".join(SYMBOL_SEARCH_PATHS)
        cmd += ["-ex", f"set solib-search-path {search_path}"]
    cmd += ["-ex", "thread apply all bt"]

    if DEBUG_GDB:
        print(f"[DEBUG] gdb cmd: {' '.join(cmd)}", file=sys.stderr)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        if DEBUG_GDB:
            print(f"[DEBUG] gdb error: {e}", file=sys.stderr)
        return {}

    raw_output = result.stdout
    if DEBUG_GDB:
        print(f"[DEBUG] gdb raw output ({len(raw_output)} bytes) stderr ({len(result.stderr)} bytes)",
              file=sys.stderr)
        # 打印前 3000 字符和后 1000 字符
        if len(raw_output) > 4000:
            print(f"[DEBUG] --- gdb HEAD (first 3000 chars) ---\n{raw_output[:3000]}", file=sys.stderr)
            print(f"[DEBUG] --- gdb TAIL (last 1000 chars) ---\n{raw_output[-1000:]}", file=sys.stderr)
        else:
            print(f"[DEBUG] --- gdb output ---\n{raw_output}", file=sys.stderr)

    parsed = _parse_gdb_all_threads_output(raw_output)

    if DEBUG_GDB:
        print(f"[DEBUG] parsed {len(parsed)} threads from GDB output", file=sys.stderr)
        for tid, stack in sorted(parsed.items()):
            # 只展示每线程的前 2 行栈帧
            head = "\n".join(stack.split("\n")[:2])
            print(f"[DEBUG]   TID {tid}: {head}", file=sys.stderr)

    return parsed


# 匹配 GDB 'thread apply all bt' 中线程头行，多种格式:
#   Thread 1 (Thread 0x... (LWP 123) "name"):
#   * 1    Thread 0x... (LWP 123) "name":
#   Thread 1 (LWP 123):
_GDB_THREAD_LWP = re.compile(r"^\*?\s*Thread\s+\d+.*\(LWP\s+(\d+)\)")


def _parse_gdb_all_threads_output(output: str) -> Dict[int, str]:
    """解析 'thread apply all bt' 的输出，按 TID (LWP) 分组栈帧。

    输入格式:
        Thread 1 (Thread 0x... (LWP 123)):
        #0  ...
        Thread 2 (Thread 0x... (LWP 456)):
        #0  ...
    """
    tid_stacks: Dict[int, str] = {}
    current_tid: Optional[int] = None
    current_lines: List[str] = []
    unmatched_count = 0

    for line in output.split("\n"):
        m = _GDB_THREAD_LWP.match(line)
        if m:
            if current_tid is not None:
                tid_stacks[current_tid] = "\n".join(current_lines)
            current_tid = int(m.group(1))
            current_lines = [line]
        elif current_tid is not None:
            current_lines.append(line)
        elif line.strip() and not line.startswith("[") and "Thread" not in line:
            unmatched_count += 1

    if current_tid is not None:
        tid_stacks[current_tid] = "\n".join(current_lines)

    if DEBUG_GDB and unmatched_count > 0:
        print(f"[DEBUG] {unmatched_count} lines before first thread header (skipped)",
              file=sys.stderr)

    return tid_stacks


def check_packages_in_stack(stack: str, packages: List[str]) -> List[str]:
    """在 GDB 栈帧中查找匹配的 package 名称。

    匹配规则: 栈帧路径或函数名中包含 package 名称。"""
    found = []
    stack_lower = stack.lower()
    for pkg in packages:
        if pkg.lower() in stack_lower:
            found.append(pkg)
    return found


# =========================================================================
# 树形结构
# =========================================================================

class Node:
    def __init__(self, is_process: bool, pid: int, tid: int,
                 name: str, current_cpu: Optional[int],
                 affinity: Set[int], packages: List[str]):
        self.is_process = is_process
        self.pid = pid
        self.tid = tid
        self.name = name
        self.current_cpu = current_cpu
        self.affinity = affinity
        self.packages = packages
        self.children: List[Node] = []

    @property
    def id(self) -> int:
        return self.tid

    def is_main_thread(self) -> bool:
        return self.tid == self.pid

    def label(self) -> str:
        t = "Process" if self.is_process else "Thread"
        main = " (Main)" if self.is_main_thread() else ""
        cpu_str = f"[{self.current_cpu}]" if self.current_cpu is not None else "[?]"
        aff_str = fmt_affinity(self.affinity)
        line = (f"[{t}] PID: {self.pid}" if self.is_process else
                f"[{t}] TID: {self.tid}{main}") + \
               f" | Name: {self.name} | CPU Affinity: {cpu_str} {aff_str}"
        if self.packages:
            for pkg in self.packages:
                line += f" -> Package: {pkg}"
        return line


def build_tree(pid: int, packages: List[str],
               visited: Optional[Set[int]] = None) -> Node:
    """递归构建进程树。

    避免循环: visited 集合记录已处理的 PID。"""
    if visited is None:
        visited = set()
    if pid in visited:
        return Node(True, pid, pid, "(cycle)", None, set(), [])
    visited.add(pid)

    tids = get_all_tids(pid)
    if not tids:
        return Node(True, pid, pid, "(gone)", None, set(), [])

    # 一次 GDB 获取全部线程栈（核心优化: N 线程 1 次 gdb 替代 N 次）
    all_stacks = gdb_get_all_stacks(pid)

    # 取主线程信息作为 process 节点信息
    main_name = get_tid_name(pid)
    main_affinity = get_cpu_affinity(pid)
    main_cpu = get_current_cpu(pid)
    main_stack = all_stacks.get(pid, "")
    main_pkgs = check_packages_in_stack(main_stack, packages)

    root = Node(True, pid, pid, main_name, main_cpu, main_affinity, main_pkgs)

    # 添加所有线程（包括主线程本身）
    for tid in tids:
        if tid == pid:
            thread_node = Node(False, pid, tid, main_name, main_cpu,
                               main_affinity, main_pkgs)
        else:
            t_name = get_tid_name(tid)
            t_affinity = get_cpu_affinity(tid)
            t_cpu = get_current_cpu(tid)
            t_stack = all_stacks.get(tid, "")
            t_pkgs = check_packages_in_stack(t_stack, packages)
            thread_node = Node(False, pid, tid, t_name, t_cpu,
                               t_affinity, t_pkgs)
        root.children.append(thread_node)

    # 递归处理子进程
    child_pids = get_child_pids(pid)
    for cpid in child_pids:
        if cpid not in visited:
            child_node = build_tree(cpid, packages, visited)
            root.children.append(child_node)

    return root


# =========================================================================
# 打印
# =========================================================================

def print_tree(node: Node, prefix: str = "", is_last: bool = True):
    """递归打印树形结构。"""
    if prefix == "":
        # 根节点
        print(node.label())
    else:
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{node.label()}")

    new_prefix = prefix + ("    " if is_last else "│   ")

    for i, child in enumerate(node.children):
        print_tree(child, new_prefix, i == len(node.children) - 1)


def run_tree(pid: int, packages: List[str]):
    """主流程：构建并打印树。"""
    root = build_tree(pid, packages)
    print_tree(root)


def main():
    parser = argparse.ArgumentParser(
        description="进程/线程信息查看工具",
        usage="python print_pid_info.py <pid> <package_name_0> [package_name_1 ...]",
    )
    parser.add_argument("pid", type=int, help="目标进程 PID")
    parser.add_argument("packages", nargs="+", help="要检测的 package 名称，如 mooncake zmq")
    args = parser.parse_args()

    pid = args.pid
    packages = args.packages

    # 检查进程是否存在
    if not os.path.isdir(f"/proc/{pid}"):
        print(f"Error: PID {pid} 不存在", file=sys.stderr)
        sys.exit(1)

    run_tree(pid, packages)


if __name__ == "__main__":
    main()
