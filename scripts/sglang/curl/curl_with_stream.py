"""
echo "错误：请使用 python 而不是 bash 运行此脚本！" >&2; exit 1

有什么用？
实现了一个流式curl命令，可以不断实时追加新的打印，方便及时观察到curl结果中存在的问题

如何使用？
1. 修改本文件的前面的prompt、url、max_new_tokens
2. 直接运行本脚本 python xx.py
3. 如果还有什么需要配置的，比如连接超时，就具体根据下面的代码去修改了
"""

import json
import sys
import time
import requests


p = "请介绍秦始皇派蒙"

prompt_text = p
url = "http://127.0.0.1:8880/generate"
temperature = 0
max_new_tokens = 3000


def generate_stream():
    headers = {"Content-Type": "application/json"}
    payload = {
        "text": prompt_text,
        "sampling_params": {"temperature": temperature, "max_new_tokens": max_new_tokens},
        "stream": True,
    }

    response = None
    try:
        # 5秒连接超时，15秒无数据传输超时
        response = requests.post(
            url, headers=headers, json=payload, stream=True, timeout=(5, 15)
        )
        response.raise_for_status()
    except KeyboardInterrupt:
        print("\n在连接建立前被用户终止 (Ctrl+C)。")
        return
    except Exception as e:
        print(f"建立连接请求失败: {e}", file=sys.stderr)
        return

    def iter_lines(resp):
        buffer = b""
        for chunk in resp.iter_content(chunk_size=1024):
            buffer += chunk
            while True:
                idx_null = buffer.find(b"\0")
                idx_nl = buffer.find(b"\n")
                if idx_null == -1 and idx_nl == -1:
                    break
                if idx_null != -1 and (idx_nl == -1 or idx_null < idx_nl):
                    line = buffer[:idx_null]
                    buffer = buffer[idx_null + 1 :]
                else:
                    line = buffer[:idx_nl]
                    buffer = buffer[idx_nl + 1 :]
                yield line
        if buffer:
            yield buffer

    print("开始生成:\n", flush=True)

    prev_len = 0
    last_output_ids = []  # 记录最新获取到的 output_ids
    last_meta_info = {}  # 记录最新获取到的元数据信息

    start_time = None  # 记录解码开始时间（用于计算客户端 TPS）
    end_time = None  # 记录最新的数据包接收时间

    try:
        for line in iter_lines(response):
            line_str = line.decode("utf-8").strip()
            if not line_str:
                continue

            if line_str.startswith("data:"):
                line_str = line_str[len("data:") :].strip()
            if line_str == "[DONE]":
                break

            try:
                data = json.loads(line_str)

                # 实时更新 Token 信息和元数据
                if "output_ids" in data:
                    last_output_ids = data["output_ids"]
                if "meta_info" in data:
                    last_meta_info = data["meta_info"]

                current_text = data.get("text", "")
                if isinstance(current_text, list):
                    current_text = current_text[0] if current_text else ""

                if current_text.startswith(prompt_text):
                    generated_text = current_text[len(prompt_text) :]
                else:
                    generated_text = current_text

                # 记录开始解码时间（剔除首字延迟 TTFT）
                prompt_tokens_count = last_meta_info.get("prompt_tokens", 0)
                has_tokens_generated = (
                    last_meta_info.get("completion_tokens", 0) > 0
                    or len(last_output_ids) > prompt_tokens_count
                    or len(generated_text) > 0
                )
                if start_time is None and has_tokens_generated:
                    start_time = time.time()

                # 成功解析完一个包就更新时间
                if start_time is not None:
                    end_time = time.time()

                # 增量打印文本
                if len(generated_text) > prev_len:
                    new_text = generated_text[prev_len:]
                    print(new_text, end="", flush=True)
                    prev_len = len(generated_text)

            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    except KeyboardInterrupt:
        print("\n\n[检测到 Ctrl+C，已手动终止生成]")
    except Exception as e:
        print(f"\n\n[网络异常或服务器意外中断: {e}]")
    finally:
        if response:
            response.close()

    # --- 计算和展示最终统计面板 ---
    print("\n" + "=" * 50)
    print("生成结束/终止统计信息：")

    # 1. 计算生成的 Token 数量
    prompt_tokens = last_meta_info.get("prompt_tokens", 0)
    completion_tokens = last_meta_info.get("completion_tokens", 0)

    generated_tokens = 0
    if completion_tokens > 0:
        generated_tokens = completion_tokens
    elif last_output_ids:
        # 剔除可能包含的 prompt prefix
        if len(last_output_ids) > prompt_tokens and prompt_tokens > 0:
            generated_tokens = len(last_output_ids) - prompt_tokens
        else:
            generated_tokens = len(last_output_ids)

    # 2. 计算客户端耗时与 TPS
    client_duration = 0.0
    if start_time and end_time:
        client_duration = end_time - start_time

    client_tps = 0.0
    if client_duration > 0 and generated_tokens > 0:
        client_tps = generated_tokens / client_duration

    # 3. 打印基础指标组合
    print(
        f"- 基础性能: 生成 Token 数: {generated_tokens} | 客户端耗时: {client_duration:.2f}s | 客户端 TPS: {client_tps:.2f} tok/s"
    )

    # 4. 提取并打印服务端 e2e_latency
    server_e2e = last_meta_info.get("e2e_latency")
    if server_e2e is not None:
        # 配合服务端的 e2e_latency 算一下真实物理吞吐 TPS
        server_tps = (
            (generated_tokens / server_e2e)
            if server_e2e > 0 and generated_tokens > 0
            else 0.0
        )
        server_tps_str = f" ({server_tps:.2f} tok/s)" if server_tps > 0 else ""
        print(f"- 服务端时延: e2e_latency: {server_e2e:.3f} 秒{server_tps_str}")

    # 5. 提取并合并打印投机解码指标 (接受率、接受长度)
    spec_rate = last_meta_info.get("spec_accept_rate")
    spec_len = last_meta_info.get("spec_accept_length")
    if spec_rate is not None or spec_len is not None:
        rate_str = f"{spec_rate * 100:.2f}%" if spec_rate is not None else "N/A"
        len_str = f"{spec_len:.3f}" if spec_len is not None else "N/A"
        print(f"- 投机性能: 接受率 (Rate): {rate_str} | 平均接受长度: {len_str}")

    # 6. 提取并合并打印草稿验证数据 (接受草稿数、提议草稿数、验证次数)
    accepted_drafts = last_meta_info.get(
        "spec_accepted_drafts"
    ) or last_meta_info.get("spec_num_correct_drafts")
    proposed_drafts = last_meta_info.get(
        "spec_proposed_drafts"
    ) or last_meta_info.get("spec_num_proposed_drafts")
    verify_ct = last_meta_info.get("spec_verify_ct")

    if accepted_drafts is not None or proposed_drafts is not None:
        acc_val = accepted_drafts if accepted_drafts is not None else "-"
        prop_val = proposed_drafts if proposed_drafts is not None else "-"
        verify_str = (
            f" | 验证次数 (Verify Ct): {verify_ct}"
            if verify_ct is not None
            else ""
        )
        print(f"- 草稿数据: 接受/提议草稿数: {acc_val} / {prop_val}{verify_str}")

    # 7. 打印其他辅助信息
    if prompt_tokens > 0:
        print(f"- 其它指标: 输入 Prompt Tokens 数: {prompt_tokens}")

    print("=" * 50)


if __name__ == "__main__":
    generate_stream()
