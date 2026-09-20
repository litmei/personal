#!/usr/bin/env python3

import argparse
import asyncio
import random
import time
from statistics import mean

import aiohttp


URL = "http://127.0.0.1:8880/generate"
TEXT = "介绍下秦始皇"

# 并发数
CONCURRENCY = 32

# max_new_tokens 随机范围 [MIN, MAX]
MIN_NEW_TOKENS = 20
MAX_NEW_TOKENS = 512

# 总请求数
TOTAL_REQUESTS = 1000

# 单个请求超时时间
TIMEOUT = 300


async def send_request(
    session: aiohttp.ClientSession,
    request_id: int,
    semaphore: asyncio.Semaphore,
):
    async with semaphore:
        max_new_tokens = random.randint(
            MIN_NEW_TOKENS,
            MAX_NEW_TOKENS,
        )

        payload = {
            "text": TEXT,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
            },
        }

        start = time.perf_counter()

        try:
            async with session.post(URL, json=payload) as response:
                result = await response.text()

                elapsed = time.perf_counter() - start

                if response.status == 200:
                    print(
                        f"[OK] request={request_id:5d} "
                        f"max_new_tokens={max_new_tokens:4d} "
                        f"latency={elapsed:.3f}s"
                    )

                    return {
                        "success": True,
                        "latency": elapsed,
                        "max_new_tokens": max_new_tokens,
                    }

                print(
                    f"[ERROR] request={request_id:5d} "
                    f"status={response.status} "
                    f"max_new_tokens={max_new_tokens} "
                    f"latency={elapsed:.3f}s "
                    f"response={result[:200]}"
                )

                return {
                    "success": False,
                    "latency": elapsed,
                    "max_new_tokens": max_new_tokens,
                }

        except Exception as e:
            elapsed = time.perf_counter() - start

            print(
                f"[EXCEPTION] request={request_id:5d} "
                f"max_new_tokens={max_new_tokens} "
                f"latency={elapsed:.3f}s "
                f"error={e}"
            )

            return {
                "success": False,
                "latency": elapsed,
                "max_new_tokens": max_new_tokens,
            }


async def main():
    semaphore = asyncio.Semaphore(CONCURRENCY)

    timeout = aiohttp.ClientTimeout(total=TIMEOUT)

    connector = aiohttp.TCPConnector(
        limit=CONCURRENCY,
        limit_per_host=CONCURRENCY,
    )

    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
    ) as session:

        start_time = time.perf_counter()

        tasks = [
            asyncio.create_task(
                send_request(
                    session,
                    request_id=i,
                    semaphore=semaphore,
                )
            )
            for i in range(TOTAL_REQUESTS)
        ]

        results = await asyncio.gather(*tasks)

        total_time = time.perf_counter() - start_time

    # =========================
    # 统计
    # =========================

    success = [
        x for x in results
        if x["success"]
    ]

    failed = [
        x for x in results
        if not x["success"]
    ]

    latencies = sorted(
        x["latency"]
        for x in success
    )

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"Total requests : {TOTAL_REQUESTS}")
    print(f"Success        : {len(success)}")
    print(f"Failed         : {len(failed)}")
    print(f"Concurrency    : {CONCURRENCY}")
    print(
        f"max_new_tokens : "
        f"[{MIN_NEW_TOKENS}, {MAX_NEW_TOKENS}]"
    )
    print(f"Total time     : {total_time:.3f}s")

    if success:
        print(f"QPS            : {len(success) / total_time:.2f}")
        print(f"Avg latency    : {mean(latencies):.3f}s")

        def percentile(data, p):
            index = int(len(data) * p / 100)
            index = min(index, len(data) - 1)
            return data[index]

        print(f"P50 latency    : {percentile(latencies, 50):.3f}s")
        print(f"P90 latency    : {percentile(latencies, 90):.3f}s")
        print(f"P95 latency    : {percentile(latencies, 95):.3f}s")
        print(f"P99 latency    : {percentile(latencies, 99):.3f}s")

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())