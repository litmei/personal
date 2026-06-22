完整脚本：

```bash
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=10
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
export SGLANG_SET_CPU_AFFINITY=1

MODEL_PATH=/path/to/Kimi-K2.6-w4a8
DRAFT_MODEL_PATH=/path/to/kimi-k2.6-eagle3

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export HCCL_SOCKET_IFNAME=enp196s0f0
export GLOO_SOCKET_IFNAME=enp196s0f0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=96
export HCCL_BUFFSIZE=1200
export HCCL_OP_EXPANSION_MODE=AIV
export SGLANG_NPU_USE_MLAPO=1
export SGLANG_NPU_USE_MULTI_STREAM=1
# export TASK_QUEUE_ENABLE=0

export PYTHONPATH=/path/to/sglang/python:$PYTHONPATH


SGLANG_PREFILL_DEBUG=1 \
sglang serve \
    --model-path $MODEL_PATH \
    --trust-remote-code \
    --attention-backend ascend \
    --device npu \
    --quantization modelslim \
    --dtype bfloat16 \
    --tp-size 16 \
    --mem-fraction-static 0.753 \
    --max-running-requests 80 \
    --chunked-prefill-size 32768 \
    --context-length 6144 \
    --max-prefill-tokens 65536 \
    --enable-multimodal \
    --mm-attention-backend ascend_attn \
    --sampling-backend ascend \
    --enable-dp-attention \
    --dp-size 16 \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --cuda-graph-bs-decode 1 2 3 4 5 \
    --disable-radix-cache \
    --model-loader-extra-config '{"enable_multithread_load": true}' \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path $DRAFT_MODEL_PATH \
    --speculative-num-steps 4 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 5 \
    --speculative-draft-model-quantization unquant \
    --device npu \
    --host 127.0.0.1 \
    --port 8880 \
    --prefill-delayer-max-delay-passes 200 \
    --enable-prefill-delayer
```

测试脚本：

```bash
export PYTHONPATH=/path/to/sglang/python:$PYTHONPATH

python3 -m sglang.bench_serving \
        --dataset-name random \
        --backend sglang \
        --model /path/to/Kimi-K2.6-w4a8/ \
        --tokenizer /path/to/Kimi-K2.6-w4a8/ \
        --random-input-len 3500 \
        --random-output-len 1500 \
        --random-range-ratio 1.0 \
        --max-concurrency 64 \
        --num-prompts 256 \
        --base-url http://127.0.0.1:8880 \
        --dataset-path /path/to/ShareGPT_V3/ShareGPT_V3_unfiltered_cleaned_split.json \
        --warmup-requests 0
```

性能数据：

```bash
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 64        
Successful requests:                     256       
Benchmark duration (s):                  191.70    
Total input tokens:                      896000    
Total input text tokens:                 896000    
Total generated tokens:                  382536    
Total generated tokens (retokenized):    381034    
Request throughput (req/s):              1.34      
Input token throughput (tok/s):          4673.97   
Output token throughput (tok/s):         1995.49   
Peak output token throughput (tok/s):    3609.00   
Peak concurrent requests:                83        
Total token throughput (tok/s):          6669.46   
Concurrency:                             54.80     
Accept length:                           4.21      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   41036.99  
Median E2E Latency (ms):                 38959.23  
P90 E2E Latency (ms):                    51320.20  
P99 E2E Latency (ms):                    62931.26  
---------------Time to First Token----------------
Mean TTFT (ms):                          9859.24   
Median TTFT (ms):                        8859.69   
P99 TTFT (ms):                           27980.56  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          20.95     
Median TPOT (ms):                        20.13     
P99 TPOT (ms):                           40.15     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           20.94     
Median ITL (ms):                         16.38     
P95 ITL (ms):                            27.04     
P99 ITL (ms):                            223.17    
Max ITL (ms):                            2854.33   
==================================================
```

---

kimi2.7

```bash

============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 64        
Successful requests:                     256       
Benchmark duration (s):                  192.03    
Total input tokens:                      896000    
Total input text tokens:                 896000    
Total generated tokens:                  384000    
Total generated tokens (retokenized):    383935    
Request throughput (req/s):              1.33      
Input token throughput (tok/s):          4665.94   
Output token throughput (tok/s):         1999.69   
Peak output token throughput (tok/s):    3623.00   
Peak concurrent requests:                87        
Total token throughput (tok/s):          6665.63   
Concurrency:                             56.16     
Accept length:                           4.56      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   42123.49  
Median E2E Latency (ms):                 40376.83  
P90 E2E Latency (ms):                    51925.56  
P95 E2E Latency (ms):                    56832.33  
P99 E2E Latency (ms):                    70293.64  
---------------Time to First Token----------------
Mean TTFT (ms):                          9499.99   
Median TTFT (ms):                        8922.12   
P90 TTFT (ms):                           15405.22  
P95 TTFT (ms):                           18984.28  
P99 TTFT (ms):                           23021.88  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          21.76     
Median TPOT (ms):                        20.69     
P90 TPOT (ms):                           25.30     
P95 TPOT (ms):                           29.86     
P99 TPOT (ms):                           40.82     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           21.76     
Median ITL (ms):                         16.50     
P90 ITL (ms):                            20.68     
P95 ITL (ms):                            28.10     
P99 ITL (ms):                            223.29    
Max ITL (ms):                            3909.99   
==================================================
```