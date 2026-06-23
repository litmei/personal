修改项：--mem-fraction-static 0.82

完整脚本：

```bash
# System Settings hello
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=10
sysctl -w kernel.numa_balancing=0

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_SET_CPU_AFFINITY=1
export STREAMS_PER_DEVICE=32

export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
 
P_IP=(<'ip0'>)
D_IP=(<'ip1'>)

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`

export ASCEND_MF_STORE_URL="tcp://${P_IP[0]}:24669"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=60

MODEL_PATH=/home/weights/Kimi-K2.6-w4a8
DRAFT_MODEL_PATH=/home/weights/kimi-k2.6-eagle3

for i in "${!P_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${P_IP[$i]}" || "$LOCAL_HOST2" == "${P_IP[$i]}" ]];
    then
        echo "Prefill -> ${P_IP[$i]}"

        export HCCL_SOCKET_IFNAME=lo
        export GLOO_SOCKET_IFNAME=lo

        export HCCL_BUFFSIZE=1800

        python -m sglang.launch_server \
            --model-path ${MODEL_PATH} --quantization modelslim --dtype bfloat16 \
            --disaggregation-mode prefill --disaggregation-transfer-backend ascend \
            --host ${P_IP[$i]} --port 8100 --disaggregation-bootstrap-port $((8998+$i)) --nnodes 1 --node-rank 0 \
            --trust-remote-code --device npu --attention-backend ascend \
            --tp-size 16  \
            --mem-fraction-static 0.78 --max-running-requests 2 \
            --moe-a2a-backend deepep --deepep-mode auto \
            --chunked-prefill-size 16384 --prefill-max-requests 2 --max-prefill-tokens 65536 \
            --enable-multimodal --mm-attention-backend ascend_attn --sampling-backend ascend

        exit 1
    fi
done


for i in "${!D_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];
    then
        echo "Decode -> ${D_IP[$i]}"

        export HCCL_SOCKET_IFNAME=lo
        export GLOO_SOCKET_IFNAME=lo

        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
        export HCCL_BUFFSIZE=1200

        export SGLANG_ENABLE_SPEC_V2=1
        export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

        # npu acceleration operator
        export SGLANG_NPU_USE_MLAPO=1
        export SGLANG_NPU_USE_MULTI_STREAM=1

        python -m sglang.launch_server \
            --model-path ${MODEL_PATH} --quantization modelslim --dtype bfloat16 \
            --disaggregation-mode decode --disaggregation-transfer-backend ascend \
            --host ${D_IP[$i]} --port 8111 --dist-init-addr ${D_IP[0]}:5000 --nnodes 1 --node-rank $i \
            --trust-remote-code --device npu --attention-backend ascend \
            --tp-size 16 --mem-fraction-static 0.82 --max-running-requests 2 \
            --enable-dp-attention --dp-size 2 --enable-dp-lm-head \
            --disable-radix-cache \
            --enable-multimodal --mm-attention-backend ascend_attn --sampling-backend ascend \
            --moe-a2a-backend deepep --deepep-mode auto \
            --cuda-graph-bs 1 2 4 6 8 \
            --speculative-algorithm EAGLE3 \
            --speculative-draft-model-path $DRAFT_MODEL_PATH \
            --speculative-num-steps 4 \
            --speculative-eagle-topk 1 \
            --speculative-num-draft-tokens 5 \
            --speculative-draft-model-quantization unquant
      exit 1
    fi
done
```

测试脚本：

```bash
export PYTHONPATH=/path/to/sglang/python:$PYTHONPATH

# 清理缓存
curl -s -X POST "http://127.0.0.1:8880/flush_cache?timeout=30"

# 进行缓存
python3 -m sglang.bench_serving \
        --dataset-name generated-shared-prefix \
        --backend sglang \
        --model /home/weights/Kimi-K2.6-w4a8/ \
        --gsp-num-groups 1 \
        --gsp-prompts-per-group 2 \
        --gsp-system-prompt-len 57600 \
        --gsp-question-len 0 \
        --gsp-output-len 1 \
        --max-concurrency 2 \
        --base-url http://127.0.0.1:8880 \
        --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
        --warmup-requests 0 \
        --gsp-range-ratio 1.0

# 实际测试
python3 -m sglang.bench_serving \
        --dataset-name generated-shared-prefix \
        --backend sglang \
        --model /home/weights/Kimi-K2.6-w4a8/ \
        --gsp-num-groups 1 \
        --gsp-prompts-per-group 8 \
        --gsp-system-prompt-len 57600 \
        --gsp-question-len 6400 \
        --gsp-output-len 1500 \
        --max-concurrency 2 \
        --base-url http://127.0.0.1:8880 \
        --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
        --warmup-requests 0 \
        --gsp-range-ratio 1.0

```

性能数据：

```bash
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 2         
Successful requests:                     8         
Benchmark duration (s):                  77.73     
Total input tokens:                      523613    
Total input text tokens:                 523613    
Total generated tokens:                  12000     
Total generated tokens (retokenized):    12000     
Request throughput (req/s):              0.10      
Input token throughput (tok/s):          6736.48   
Output token throughput (tok/s):         154.38    
Peak output token throughput (tok/s):    185.00    
Peak concurrent requests:                4         
Total token throughput (tok/s):          6890.86   
Concurrency:                             1.99      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   19376.11  
Median E2E Latency (ms):                 19372.69  
P90 E2E Latency (ms):                    20366.67  
P99 E2E Latency (ms):                    21778.76  
---------------Time to First Token----------------
Mean TTFT (ms):                          2481.94   
Median TTFT (ms):                        2740.08   
P99 TTFT (ms):                           3118.77   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          11.27     
Median TPOT (ms):                        10.99     
P99 TPOT (ms):                           12.49     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           11.27     
Median ITL (ms):                         10.90     
P95 ITL (ms):                            11.95     
P99 ITL (ms):                            18.61     
Max ITL (ms):                            98.35     
==================================================
```


---

kimi27

``` 
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 2         
Successful requests:                     8         
Benchmark duration (s):                  80.40     
Total input tokens:                      523613    
Total input text tokens:                 523613    
Total generated tokens:                  12000     
Total generated tokens (retokenized):    12000     
Request throughput (req/s):              0.10      
Input token throughput (tok/s):          6512.68   
Output token throughput (tok/s):         149.26    
Peak output token throughput (tok/s):    182.00    
Peak concurrent requests:                4         
Total token throughput (tok/s):          6661.93   
Concurrency:                             2.00      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   20063.27  
Median E2E Latency (ms):                 19994.52  
P90 E2E Latency (ms):                    21000.01  
P99 E2E Latency (ms):                    21289.69  
---------------Time to First Token----------------
Mean TTFT (ms):                          2669.17   
Median TTFT (ms):                        2992.19   
P99 TTFT (ms):                           3074.00   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          11.60     
Median TPOT (ms):                        11.46     
P99 TPOT (ms):                           12.16     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           11.61     
Median ITL (ms):                         11.19     
P95 ITL (ms):                            11.96     
P99 ITL (ms):                            18.74     
Max ITL (ms):                            113.74    
==================================================
```