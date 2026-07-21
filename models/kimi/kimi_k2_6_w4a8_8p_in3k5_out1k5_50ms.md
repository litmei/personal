修改项：--mem-fraction-static 0.895

---

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
# export TASK_QUEUE_ENABLE=0

#export PYTHONPATH=/path/to/sglang/python:$PYTHONPATH


sglang serve \
    --model-path $MODEL_PATH \
    --trust-remote-code \
    --attention-backend ascend \
    --device npu \
    --quantization modelslim \
    --dtype bfloat16 \
    --tp-size 16 \
    --mem-fraction-static 0.895 \
    --max-running-requests 208 \
    --chunked-prefill-size 32768 \
    --context-length 6144 \
    --max-prefill-tokens 16384 \
    --enable-multimodal \
    --mm-attention-backend ascend_attn \
    --sampling-backend ascend \
    --enable-dp-attention \
    --dp-size 16 \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --cuda-graph-bs-decode 1 2 4 8 12 13 \
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
        --random-input-len 3500 \
        --random-output-len 1500 \
        --random-range-ratio 1.0 \
        --max-concurrency 192 \
        --num-prompts 768 \
        --base-url http://127.0.0.1:8880 \
        --dataset-path /path/to/ShareGPT_V3/ShareGPT_V3_unfiltered_cleaned_split.json \
        --disable-ignore-eos \
        --warmup-requests 0
```

性能数据：

```bash
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 192       
Successful requests:                     768       
Benchmark duration (s):                  328.71    
Total input tokens:                      2688000   
Total input text tokens:                 2688000   
Total generated tokens:                  1147842   
Total generated tokens (retokenized):    1144836   
Request throughput (req/s):              2.34      
Input token throughput (tok/s):          8177.48   
Output token throughput (tok/s):         3491.98   
Peak output token throughput (tok/s):    7206.00   
Peak concurrent requests:                238       
Total token throughput (tok/s):          11669.46  
Concurrency:                             166.91    
Accept length:                           4.60      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   71438.54  
Median E2E Latency (ms):                 69042.03  
P90 E2E Latency (ms):                    88258.84  
P99 E2E Latency (ms):                    116196.65 
---------------Time to First Token----------------
Mean TTFT (ms):                          12912.51  
Median TTFT (ms):                        11414.29  
P99 TTFT (ms):                           38956.13  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          39.36     
Median TPOT (ms):                        39.38     
P99 TPOT (ms):                           71.14     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           39.28     
Median ITL (ms):                         25.71     
P95 ITL (ms):                            64.22     
P99 ITL (ms):                            268.73    
Max ITL (ms):                            12173.87  
==================================================
```


---

kimi2.7

```bash
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 192       
Successful requests:                     768       
Benchmark duration (s):                  348.86    
Total input tokens:                      2688000   
Total input text tokens:                 2688000   
Total generated tokens:                  1152000   
Total generated tokens (retokenized):    1151762   
Request throughput (req/s):              2.20      
Input token throughput (tok/s):          7705.06   
Output token throughput (tok/s):         3302.17   
Peak output token throughput (tok/s):    6836.00   
Peak concurrent requests:                230       
Total token throughput (tok/s):          11007.22  
Concurrency:                             170.85    
Accept length:                           4.55      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   77607.46  
Median E2E Latency (ms):                 74925.60  
P90 E2E Latency (ms):                    97021.59  
P95 E2E Latency (ms):                    109325.60 
P99 E2E Latency (ms):                    134470.17 
---------------Time to First Token----------------
Mean TTFT (ms):                          14256.83  
Median TTFT (ms):                        11817.90  
P90 TTFT (ms):                           24669.15  
P95 TTFT (ms):                           25936.86  
P99 TTFT (ms):                           58436.37  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          42.26     
Median TPOT (ms):                        41.22     
P90 TPOT (ms):                           52.17     
P95 TPOT (ms):                           59.14     
P99 TPOT (ms):                           79.00     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           42.26     
Median ITL (ms):                         25.68     
P90 ITL (ms):                            35.65     
P95 ITL (ms):                            68.27     
P99 ITL (ms):                            298.98    
Max ITL (ms):                            22450.95  
==================================================
```