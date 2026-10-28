#!/usr/bin/env python3
""":"
echo "错误：请使用 python 而不是 bash 运行此脚本！" >&2; exit 1
"""
"""
批量 benchmark 脚本（流式，单请求独立计时）

功能
1. 读取 tokenizer，构造指定 token 长度的输入文本（长度按 tokenizer 实际编码长度校准）
2. 支持生成公共前缀（--prefix-len）：所有用例共享同一段前缀，再拼接各自随机后缀凑满目标长度，
   用于验证 RadixAttention / prefix cache 的命中效果
3. 组 batch：一批内并发发出多个独立请求，每个用例的输入长度可独立指定
4. 预发送用例预览（--preview），并给出公共前缀与每个用例输入文本的 sha256
5. 可控制最大输出长度、温度
6. 打印每个用例的输出文本及其 sha256（--print-output）
7. 打印结果汇总表格：输入长度 / 输入SHA / 输出长度 / MTP命中率 / TTFT / TPOT / 总耗时

用法示例
  # 一批 3 条，输入长度分别 128 / 1024 / 4096，最大输出 256，温度 0
  python curl_bench_batch.py \
      --tokenizer-path /path/to/tokenizer \
      --url http://127.0.0.1:8880/generate \
      --input-lens 128,1024,4096 \
      --max-new-tokens 256 \
      --temperature 0 \
      --preview --print-output

  # 一批 8 条，长度在 [256, 4096] 内随机，跑 3 轮
  python curl_bench_batch.py \
      --tokenizer-path /path/to/tokenizer \
      --batch-size 8 --input-len-min 256 --input-len-max 4096 \
      --max-new-tokens 512 --repeat 3

  # 公共前缀 1024 token，一批 4 条，总长度分别 2048 / 4096 / 8192 / 16384
  python curl_bench_batch.py \
      --tokenizer-path /path/to/tokenizer \
      --prefix-len 1024 --input-lens 2048,4096,8192,16384 \
      --max-new-tokens 128 --preview

  # 一批 3 条（并发 3 条）
  python curl_bench_batch.py --tokenizer-path /path/to/tokenizer \
      --input-lens "4096,4096,4096" --max-new-tokens 128

  # 3 批、每批 1 条，批间顺序执行（等价于不组 batch 地连发 3 次）
  python curl_bench_batch.py --tokenizer-path /path/to/tokenizer \
      --input-lens "4096 4096 4096" --max-new-tokens 128

  # 先发一批 3 条，再顺序发 3 批各 1 条
  python curl_bench_batch.py --tokenizer-path /path/to/tokenizer \
      --input-lens "4096,4096,4096 4096 4096 4096" --max-new-tokens 128

  # 公共前缀 1024 token，一批 4 条，非前缀部分分别 1024 / 3072 / 7168 / 15360
  #（实际总长度 = 1024 + 非前缀长度）
  python curl_bench_batch.py \
      --tokenizer-path /path/to/tokenizer \
      --prefix-len 1024 --input-lens 1024,3072,7168,15360 \
      --max-new-tokens 128 --preview

说明
- --input-lens 表示**非公共前缀部分**的 token 长度（不给 --prefix-len 时即总长度）；
  实际总长度 = prefix_len + input_len，表格中的“输入长度”是实际总长度。
- --input-lens 语法：同一批内用逗号分隔，不同批之间用空格分隔，
  如 "4096,4096,4096 4096" = 一批 3 条 + 一批 1 条；批内并发、批间顺序执行。
- 首次发送前建议服务端已完成 warmup，否则首批 TTFT 会偏高。
- 长度以 tokenizer.encode(text, add_special_tokens=False) 的真实编码为准，
  若与目标长度无法完全对齐，预览里会打印实际值。
- 公共前缀在同一次运行内（跨批、跨 --repeat 轮）保持不变；不同次运行默认使用随机种子，
  前缀会变化（--prefix-seed N 可复现某次运行的前缀，preview/总览会打印该 seed）。
- 未给 --input-lens 时，用 --batch-size 随机采样长度并组成**一批**；两者互斥。
- MTP 命中率优先取 meta_info.spec_accept_rate；否则用 accepted/proposed drafts 计算。
"""

import argparse
import asyncio
import hashlib
import json
import random
import sys
import time
import unicodedata
from statistics import mean

import aiohttp

# 用于拼装指定长度输入的语料（中英混排，避免单字符重复导致 tokenizer 合并异常）
CORPUS = (
    "SGLang 是一个面向大语言模型与多模态模型的高性能推理框架，支持 RadixAttention 前缀缓存、"
    "连续批处理（continuous batching）、投机解码与多卡并行。The quick brown fox jumps over the lazy dog. "
    "在长上下文场景下，KV cache 的显存占用、注意力后端的选择以及 page size 都会显著影响吞吐与首字时延。"
    "Benchmark 时应当固定随机种子、关闭采样噪声，并保证每个用例的输入长度可复现。"
    "大语言模型的推理性能通常用 TTFT（time to first token）、TPOT（time per output token）"
    "以及端到端时延来衡量，同时还需要关注投机解码的接受率与接受长度。"
)


# =========================
# 工具函数
# =========================
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def char_width(text: str) -> int:
    """终端显示宽度（中文按 2 列计算）"""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in text)


def pad(text: str, width: int, align: str = "right") -> str:
    space = max(0, width - char_width(text))
    if align == "left":
        return text + " " * space
    if align == "center":
        left = space // 2
        return " " * left + text + " " * (space - left)
    return " " * space + text


def _sample_ids(corpus_ids, length: int, rng: random.Random):
    """从语料中随机起点循环取样，得到 length 个 token id"""
    start = rng.randrange(len(corpus_ids))
    return (corpus_ids[start:] + corpus_ids * (length // len(corpus_ids) + 2))[:length]


def build_prompt_text(tokenizer, unique_len: int, rng: random.Random, prefix_text: str = "") -> str:
    """构造 prefix_text + 随机后缀 的文本，其中**非前缀部分**的 token 数尽量等于 unique_len

    prefix_text 为空时即构造总长 unique_len 的纯随机文本。
    """
    corpus_ids = tokenizer.encode(CORPUS, add_special_tokens=False)
    if not corpus_ids:
        raise RuntimeError("语料 tokenize 结果为空，请检查 tokenizer 路径")

    prefix_len = len(tokenizer.encode(prefix_text, add_special_tokens=False)) if prefix_text else 0
    unique_len = max(1, unique_len)
    target_len = prefix_len + unique_len

    ids = _sample_ids(corpus_ids, unique_len, rng)
    text = prefix_text + tokenizer.decode(ids)

    # decode -> encode 可能因合并规则（含前后缀拼接处）产生偏差，迭代校准后缀
    for _ in range(10):
        actual = len(tokenizer.encode(text, add_special_tokens=False))
        delta = target_len - actual
        if delta == 0:
            break
        if delta > 0:
            ids = ids + corpus_ids[:delta]
        else:
            ids = ids[: max(1, len(ids) + delta)]
        text = prefix_text + tokenizer.decode(ids)

    return text


def build_shared_prefix(tokenizer, prefix_len: int, seed: int) -> str:
    """生成所有用例共享的公共前缀，token 长度等于 prefix_len"""
    if prefix_len <= 0:
        return ""
    rng = random.Random(seed + 1)  # 与后缀采样种子错开，互不影响
    return build_prompt_text(tokenizer, prefix_len, rng)


def resolve_prefix_seed(args):
    """未指定 --prefix-seed 时，每次运行随机取一个种子，保证跨次运行前缀会变化"""
    if args.prefix_len <= 0:
        return None
    if args.prefix_seed is not None:
        return args.prefix_seed
    return random.SystemRandom().randrange(1 << 32)


def mtp_hit_rate(meta: dict):
    """MTP / 投机解码命中率"""
    rate = meta.get("spec_accept_rate")
    if rate is not None:
        return rate
    accepted = meta.get("spec_accepted_drafts")
    if accepted is None:
        accepted = meta.get("spec_num_correct_drafts")
    proposed = meta.get("spec_proposed_drafts")
    if proposed is None:
        proposed = meta.get("spec_num_proposed_drafts")
    if accepted is not None and proposed:
        return accepted / proposed
    return None


async def iter_sse_lines(response):
    """把流式响应切成行（兼容 \\n 与 \\0 分隔）"""
    buffer = b""
    async for chunk in response.content.iter_any():
        buffer += chunk
        while True:
            idx_n = buffer.find(b"\n")
            idx_z = buffer.find(b"\0")
            if idx_n == -1 and idx_z == -1:
                break
            if idx_z == -1 or (idx_n != -1 and idx_n < idx_z):
                idx = idx_n
            else:
                idx = idx_z
            line = buffer[:idx].decode("utf-8", errors="ignore").strip()
            buffer = buffer[idx + 1:]
            if line:
                yield line

    tail = buffer.decode("utf-8", errors="ignore").strip()
    if tail:
        yield tail


# =========================
# 单请求
# =========================
async def run_one(session, url, case, args):
    payload = {
        "text": case["prompt"],
        "sampling_params": {
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "ignore_eos": bool(args.ignore_eos),
        },
        "stream": True,
    }

    result = {
        "case_id": case["case_id"],
        "batch": case["batch"],
        "prompt_len": case["target_len"],
        "prompt_sha": case["prompt_sha"],
        "output_text": "",
        "output_len": 0,
        "output_sha": "",
        "ttft": None,
        "tpot": None,
        "elapsed": None,
        "mtp_rate": None,
        "accept_len": None,
        "success": False,
        "error": "",
    }

    start = time.perf_counter()
    ttft = None
    output_text = ""
    last_meta = {}
    last_output_ids = []

    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                body = await response.text()
                result["elapsed"] = time.perf_counter() - start
                result["error"] = f"HTTP {response.status}: {body[:300]}"
                return result

            async for line in iter_sse_lines(response):
                if line.startswith("data:"):
                    line = line[len("data:"):].strip()
                if line == "[DONE]":
                    break

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                now = time.perf_counter()

                if isinstance(data.get("meta_info"), dict):
                    last_meta = data["meta_info"]
                if isinstance(data.get("output_ids"), list):
                    last_output_ids = data["output_ids"]

                current = data.get("text", "")
                if isinstance(current, list):
                    current = current[0] if current else ""

                if current:
                    output_text = current
                    if ttft is None:
                        ttft = now - start
                elif ttft is None and last_meta.get("completion_tokens", 0) > 0:
                    ttft = now - start

        elapsed = time.perf_counter() - start

        prompt_tokens = last_meta.get("prompt_tokens", 0) or 0
        output_len = last_meta.get("completion_tokens", 0) or 0
        if output_len <= 0 and last_output_ids:
            output_len = len(last_output_ids) - prompt_tokens if prompt_tokens else len(last_output_ids)

        result.update(
            {
                "output_text": output_text,
                "output_len": output_len,
                "output_sha": sha256_text(output_text),
                "ttft": ttft,
                "elapsed": elapsed,
                "mtp_rate": mtp_hit_rate(last_meta),
                "accept_len": last_meta.get("spec_accept_length"),
                "success": True,
            }
        )

        if ttft is not None and output_len > 1:
            result["tpot"] = (elapsed - ttft) / (output_len - 1)

        return result

    except Exception as e:
        result["elapsed"] = time.perf_counter() - start
        result["error"] = repr(e)
        return result


# =========================
# 用例构造 / 打印
# =========================
def parse_batch_lens(spec: str):
    """解析 --input-lens：逗号分隔同一批内的用例，空格分隔不同的批

    "4096,4096,4096"     -> [[4096,4096,4096]]   一批 3 条（并发）
    "4096 4096 4096"     -> [[4096],[4096],[4096]] 3 批各 1 条（顺序执行）
    "4096,4096 4096"     -> [[4096,4096],[4096]]  一批 2 条 + 一批 1 条
    """
    groups = []
    for group in spec.split():
        try:
            lens = [int(x) for x in group.replace(",", " ").split() if x]
        except ValueError:
            raise ValueError(f"无法解析用例长度 {group!r}：应为逗号分隔的整数")
        if lens:
            groups.append(lens)
    if not groups:
        raise ValueError(f"无法解析用例长度: {spec!r}")
    return groups


def make_cases(tokenizer, args):
    rng = random.Random(args.seed)

    if args.input_lens.strip():
        lens_groups = parse_batch_lens(args.input_lens)
    else:
        lens_groups = [
            [
                rng.randint(args.input_len_min, args.input_len_max)
                for _ in range(args.batch_size)
            ]
        ]

    prefix_seed = resolve_prefix_seed(args)
    prefix_text = build_shared_prefix(tokenizer, args.prefix_len, prefix_seed or 0)
    prefix_tokens = len(tokenizer.encode(prefix_text, add_special_tokens=False)) if prefix_text else 0

    cases = []
    case_id = 0
    for batch_idx, group in enumerate(lens_groups):
        for unique_len in group:
            prompt = build_prompt_text(tokenizer, unique_len, rng, prefix_text)
            actual_len = len(tokenizer.encode(prompt, add_special_tokens=False))
            cases.append(
                {
                    "case_id": case_id,
                    "batch": batch_idx,
                    "unique_len": unique_len,
                    "target_len": actual_len,
                    "prompt": prompt,
                    "prompt_sha": sha256_text(prompt),
                }
            )
            case_id += 1

    prefix_info = {
        "text": prefix_text,
        "len": prefix_tokens,
        "sha": sha256_text(prefix_text) if prefix_text else "",
        "seed": prefix_seed,
    }
    return cases, prefix_info


def preview_cases(cases, prefix_info, args):
    batch_count = len({c["batch"] for c in cases})
    print("=" * 100)
    print(f"待发送用例预览（共 {len(cases)} 条 / {batch_count} 批，url={args.url}，"
          f"max_new_tokens={args.max_new_tokens}，temperature={args.temperature}）")
    if prefix_info["len"]:
        print(f"公共前缀: {prefix_info['len']} tokens，sha256={prefix_info['sha']}，"
              f"seed={prefix_info['seed']}（用 --prefix-seed {prefix_info['seed']} 可复现该前缀）")
        text = prefix_info["text"]
        if args.preview_chars > 0:
            text = text[: args.preview_chars] + ("..." if len(prefix_info["text"]) > args.preview_chars else "")
        print(f"  prefix: {text}")
    print("=" * 100)
    last_batch = None
    for case in cases:
        if case["batch"] != last_batch:
            last_batch = case["batch"]
            print(f"--- Batch {last_batch} ---")
        print(f"[Case {case['case_id']}] 非前缀长度={case['unique_len']} "
              f"总长度={case['target_len']} sha256={case['prompt_sha']}")
        text = case["prompt"]
        if args.preview_chars > 0:
            text = text[: args.preview_chars] + ("..." if len(case["prompt"]) > args.preview_chars else "")
        print(f"  prompt: {text}")
    print("=" * 100)


def print_outputs(results, args):
    print("=" * 100)
    print("输出文本")
    print("=" * 100)
    for r in results:
        print(f"[Case {r['case_id']}] 输入长度={r['prompt_len']} "
              f"输出长度={r['output_len']} 输出sha256={r['output_sha'] or '-'}")
        if not args.print_output:
            continue
        text = r["output_text"]
        if args.output_chars > 0:
            text = text[: args.output_chars] + ("..." if len(r["output_text"]) > args.output_chars else "")
        print(f"  output: {text}")
    print("=" * 100)


def fmt(value, spec, suffix="", empty="-"):
    if value is None:
        return empty
    return f"{value:{spec}}{suffix}"


def print_table(results, args):
    headers = ["#", "批次", "状态", "输入长度", "输入SHA", "输出长度",
               "MTP命中率", "TTFT(ms)", "TPOT(ms)", "总耗时(s)"]
    rows = []
    for r in results:
        rows.append(
            [
                str(r["case_id"]),
                str(r["batch"]),
                "OK" if r["success"] else "FAIL",
                str(r["prompt_len"]),
                r["prompt_sha"][: args.sha_len],
                str(r["output_len"]),
                fmt(r["mtp_rate"] * 100 if r["mtp_rate"] is not None else None, ".2f", "%"),
                fmt(r["ttft"] * 1000 if r["ttft"] is not None else None, ".2f"),
                fmt(r["tpot"] * 1000 if r["tpot"] is not None else None, ".2f"),
                fmt(r["elapsed"], ".3f"),
            ]
        )

    aligns = ["right", "right", "center", "right", "left", "right",
              "right", "right", "right", "right"]
    widths = [char_width(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], char_width(cell))

    line = "-+-".join("-" * w for w in widths)

    print()
    print("=" * 100)
    print("结果汇总")
    print("=" * 100)
    print(" | ".join(pad(h, widths[i], aligns[i]) for i, h in enumerate(headers)))
    print(line)
    for row in rows:
        print(" | ".join(pad(cell, widths[i], aligns[i]) for i, cell in enumerate(row)))
    print("=" * 100)

    failed = [r for r in results if not r["success"]]
    if failed:
        print("失败用例：")
        for r in failed:
            print(f"  [Case {r['case_id']}] {r['error']}")


def print_summary(results, wall_time, batch_walls, prefix_info, args):
    ok = [r for r in results if r["success"]]
    total_out = sum(r["output_len"] for r in ok)
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    tpots = [r["tpot"] for r in ok if r["tpot"] is not None]
    rates = [r["mtp_rate"] for r in ok if r["mtp_rate"] is not None]
    batch_sizes = {}
    for r in results:
        batch_sizes[r["batch"]] = batch_sizes.get(r["batch"], 0) + 1

    print()
    print("总览")
    print(f"  批数 / 用例数: {len(batch_walls)} / {len(results)}")
    print(f"  成功 / 失败  : {len(ok)} / {len(results) - len(ok)}")
    print(f"  公共前缀     : {prefix_info['len']} tokens"
          + (f"（sha256={prefix_info['sha']}，seed={prefix_info['seed']}）" if prefix_info["len"] else ""))
    print(f"  总墙钟耗时   : {wall_time:.3f}s")
    if len(batch_walls) > 1:
        print("  各批耗时     : " + " | ".join(
            f"batch{b}={w:.3f}s（{batch_sizes.get(b, 0)}条）"
            for b, w in enumerate(batch_walls)
        ))
    print(f"  输入 tokens  : {sum(r['prompt_len'] for r in ok)}")
    print(f"  输出 tokens  : {total_out}")
    if wall_time > 0:
        print(f"  输出吞吐     : {total_out / wall_time:.2f} tok/s（整批）")
    if ttfts:
        print(f"  平均 TTFT    : {mean(ttfts) * 1000:.2f} ms "
              f"(min {min(ttfts) * 1000:.2f} / max {max(ttfts) * 1000:.2f})")
    if tpots:
        print(f"  平均 TPOT    : {mean(tpots) * 1000:.2f} ms "
              f"(min {min(tpots) * 1000:.2f} / max {max(tpots) * 1000:.2f})")
    if rates:
        print(f"  平均 MTP命中 : {mean(rates) * 100:.2f}%")
    print()


# =========================
# 主流程
# =========================
async def run_bench(tokenizer, args):
    cases, prefix_info = make_cases(tokenizer, args)

    if args.preview:
        preview_cases(cases, prefix_info, args)

    # 按批次分组，批内并发、批间顺序执行
    batches = {}
    for case in cases:
        batches.setdefault(case["batch"], []).append(case)
    batch_list = [batches[i] for i in sorted(batches)]

    max_batch = max(len(b) for b in batch_list)
    print(f">>> 用例数={len(cases)}，批数={len(batch_list)}（各批条数: "
          f"{', '.join(str(len(b)) for b in batch_list)}）")

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=max_batch * 2)

    async def run_batch(session, batch):
        batch_start = time.perf_counter()
        results = await asyncio.gather(
            *[run_one(session, args.url, case, args) for case in batch]
        )
        return list(results), time.perf_counter() - batch_start

    all_results = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for round_idx in range(args.repeat):
            if args.repeat > 1:
                print(f"\n>>> Round {round_idx + 1}/{args.repeat}")

            round_start = time.perf_counter()
            results = []
            batch_walls = []
            for batch_idx, batch in enumerate(batch_list):
                if len(batch_list) > 1:
                    print(f"  -> Batch {batch_idx}/{len(batch_list) - 1}：{len(batch)} 条")
                batch_results, batch_wall = await run_batch(session, batch)
                results.extend(batch_results)
                batch_walls.append(batch_wall)
            wall_time = time.perf_counter() - round_start

            if args.repeat > 1:
                for r in results:
                    r["case_id"] = f"{round_idx}-{r['case_id']}"

            print_table(results, args)
            print_summary(results, wall_time, batch_walls, prefix_info, args)

            if args.print_output:
                print_outputs(results, args)

            all_results.extend(results)

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="基于 tokenizer 构造定长输入的批量 benchmark 脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default="http://127.0.0.1:8880/generate", help="服务端 /generate 地址")
    parser.add_argument("--tokenizer-path", required=True, help="tokenizer 目录或模型目录")

    # batch 与输入长度
    parser.add_argument("--input-lens", nargs="+", default=[],
                        help="非公共前缀部分的 token 长度：同一批用逗号分隔，不同批用空格分隔，"
                             "如 4096,4096,4096 为一批3条；4096 4096 4096 为3批各1条")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="未指定 --input-lens 时的批大小（用例数，长度随机采样）；与 --input-lens 互斥")
    parser.add_argument("--input-len-min", type=int, default=128, help="随机输入长度下界")
    parser.add_argument("--input-len-max", type=int, default=2048, help="随机输入长度上界")
    parser.add_argument("--prefix-len", type=int, default=0,
                        help="公共前缀的 token 长度（0 表示不生成前缀）；所有用例共享，其后拼接随机后缀凑满目标长度")
    parser.add_argument("--prefix-seed", type=int, default=None,
                        help="公共前缀随机种子；默认每次运行随机（每次执行前缀都会变），指定后可复现")

    # 采样参数
    parser.add_argument("--max-new-tokens", type=int, default=256, help="最大输出长度")
    parser.add_argument("--temperature", type=float, default=0.0, help="温度")
    parser.add_argument("--ignore-eos", action="store_true",
                        help="忽略 EOS，强制输出到 max_new_tokens，便于对比 TPOT")

    # 运行控制
    parser.add_argument("--repeat", type=int, default=1, help="同一批用例重复发送的轮数")
    parser.add_argument("--seed", type=int, default=0, help="随机长度采样种子")
    parser.add_argument("--timeout", type=float, default=600, help="单请求超时时间（秒）")

    # 打印控制
    parser.add_argument("--preview", action="store_true", help="发送前预览用例文本")
    parser.add_argument("--preview-chars", type=int, default=200,
                        help="预览文本截断字符数，0 表示打印全文")
    parser.add_argument("--print-output", action="store_true", help="打印每条用例的输出文本")
    parser.add_argument("--output-chars", type=int, default=300,
                        help="输出文本截断字符数，0 表示打印全文")
    parser.add_argument("--sha-len", type=int, default=12, help="表格中 sha256 显示的前缀长度")

    args = parser.parse_args()
    args.input_lens = " ".join(args.input_lens)

    # --batch-size 只在随机采样模式下使用，批结构由 --input-lens 表达
    if args.input_lens.strip() and args.batch_size is not None:
        parser.error(
            "--input-lens 与 --batch-size 不能同时指定："
            "--batch-size 仅用于随机采样模式，批结构请用 --input-lens 表达"
        )
    if args.batch_size is None:
        args.batch_size = 4

    if args.input_lens.strip():
        try:
            parse_batch_lens(args.input_lens)
        except ValueError as e:
            parser.error(str(e))

    return args


def load_tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def main():
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer_path)
    asyncio.run(run_bench(tokenizer, args))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已手动终止", file=sys.stderr)
        sys.exit(130)
