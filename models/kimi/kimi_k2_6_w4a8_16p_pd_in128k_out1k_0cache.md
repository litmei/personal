D节点修改项：--mem-fraction-static 0.82

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
  
export PYTHONPATH=/path/to/sglang/python:$PYTHONPATH
P_IP=('<prefill_ip0>')
D_IP=('<decode_ip0>')
  
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
  
        export HCCL_SOCKET_IFNAME=enp196s0f0
        export GLOO_SOCKET_IFNAME=enp196s0f0
  
        export HCCL_BUFFSIZE=1800
  
        # zb
        # zbccl
        export HCCL_BUFFSIZE=8
        unset PYTORCH_NPU_ALLOC_CONF
        export SGLANG_ZBAL_LOCAL_MEM_SIZE=61184
        export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
        # zbccl if use mix alloc
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export ZBAL_NPU_ALLOC_CONF=use_vmm_for_static_memory:True
        export SGLANG_ZBAL_BOOTSTRAP_URL="tcp://192.168.25.209:24699"
        # zbccl if support graph��[m~Hneed custom pta��[m~I
        export ZBAL_ENABLE_GRAPH=1
        # pp mock send and recv
        # export ZBAL_HCCL_OP="send,recv"
  
        python -m sglang.launch_server \
            --model-path ${MODEL_PATH} --quantization modelslim --dtype bfloat16 \
            --disaggregation-mode prefill --disaggregation-transfer-backend ascend \
            --host ${P_IP[$i]} --port 8100 --disaggregation-bootstrap-port $((8998+$i)) --nnodes 1 --node-rank 0 \
            --trust-remote-code --device npu --attention-backend ascend \
            --tp-size 16 --disable-radix-cache \
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
  
        export HCCL_SOCKET_IFNAME=enp196s0f0
        export GLOO_SOCKET_IFNAME=enp196s0f0
  
        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
        export HCCL_BUFFSIZE=1200
  
        export SGLANG_ENABLE_SPEC_V2=1
        export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
  
        export SGLANG_NPU_USE_MLAPO=1
        export SGLANG_NPU_USE_MULTI_STREAM=1
  
        python -m sglang.launch_server \
            --model-path ${MODEL_PATH} --quantization modelslim --dtype bfloat16 \
            --disaggregation-mode decode --disaggregation-transfer-backend ascend \
            --host ${D_IP[$i]} --port 8111 --dist-init-addr ${D_IP[0]}:5000 --nnodes 1 --node-rank $i \
            --trust-remote-code --device npu --attention-backend ascend \
            --tp-size 16 --mem-fraction-static 0.82 --max-running-requests 2 \
            --enable-dp-attention --dp-size 1 --enable-dp-lm-head \
            --disable-radix-cache \
            --enable-multimodal --mm-attention-backend ascend_attn --sampling-backend ascend \
            --moe-a2a-backend deepep --deepep-mode auto \
            --cuda-graph-bs 1 2 4 6 8 16 \
            --speculative-algorithm EAGLE3 \
            --speculative-draft-model-path $DRAFT_MODEL_PATH \
            --speculative-num-steps 3 \
            --speculative-eagle-topk 1 \
            --speculative-num-draft-tokens 4 \
            --speculative-draft-model-quantization unquant      
  
      exit 1
    fi
done
```

router：

```bash
export PYTHONPATH=/path/to/sglang/python:$PYTHONPATH
python -m sglang_router.launch_router \
    --pd-disaggregation --policy cache_aware \
    --prefill http://<prefill_ip0>:8100 8998 \
    --decode http://<decode_ip10:8111 \
    --host 0.0.0.0 --port 8880
```

测试脚本：

```bash 
export PYTHONPATH=/home/xjw/code/sglang/python:$PYTHONPATH

python3 -m sglang.bench_serving \
	--dataset-name random \
	--backend sglang \
	--model /path/to/Kimi-K2.6-w4a8/ \
	--random-input-len 64000 \
	--random-output-len 1500 \
	--random-range-ratio 1.0 \
	--max-concurrency 1 \
	--num-prompts 1 \
	--base-url http://127.0.0.1:8880 \
	--dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
	--warmup-requests 0 \
    --request-rate inf
```

测试结果：

```bash
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 1         
Successful requests:                     1         
Benchmark duration (s):                  41.23     
Total input tokens:                      128000    
Total input text tokens:                 128000    
Total generated tokens:                  1000      
Total generated tokens (retokenized):    1000      
Request throughput (req/s):              0.02      
Input token throughput (tok/s):          3104.88   
Output token throughput (tok/s):         24.26     
Peak output token throughput (tok/s):    72.00     
Peak concurrent requests:                1         
Total token throughput (tok/s):          3129.14   
Concurrency:                             1.00      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   41221.51  
Median E2E Latency (ms):                 41221.51  
P90 E2E Latency (ms):                    41221.51  
P99 E2E Latency (ms):                    41221.51  
---------------Time to First Token----------------
Mean TTFT (ms):                          22973.26  
Median TTFT (ms):                        22973.26  
P99 TTFT (ms):                           22973.26  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          18.27     
Median TPOT (ms):                        18.27     
P99 TPOT (ms):                           18.27     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           18.27     
Median ITL (ms):                         13.30     
P95 ITL (ms):                            17.88     
P99 ITL (ms):                            52.14     
Max ITL (ms):                            490.07    
==================================================
```


---

kimi 27

``` 
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 1         
Successful requests:                     1         
Benchmark duration (s):                  37.77     
Total input tokens:                      128000    
Total input text tokens:                 128000    
Total generated tokens:                  1000      
Total generated tokens (retokenized):    1000      
Request throughput (req/s):              0.03      
Input token throughput (tok/s):          3388.99   
Output token throughput (tok/s):         26.48     
Peak output token throughput (tok/s):    71.00     
Peak concurrent requests:                1         
Total token throughput (tok/s):          3415.46   
Concurrency:                             1.00      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   37765.22  
Median E2E Latency (ms):                 37765.22  
P90 E2E Latency (ms):                    37765.22  
P99 E2E Latency (ms):                    37765.22  
---------------Time to First Token----------------
Mean TTFT (ms):                          23088.13  
Median TTFT (ms):                        23088.13  
P99 TTFT (ms):                           23088.13  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          14.69     
Median TPOT (ms):                        14.69     
P99 TPOT (ms):                           14.69     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           14.69     
Median ITL (ms):                         13.46     
P95 ITL (ms):                            18.03     
P99 ITL (ms):                            26.58     
Max ITL (ms):                            78.50     
==================================================
```