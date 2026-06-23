修改项：--mem-fraction-static 0.662 


https://github.com/Ascend/sglang/actions/runs/28007314562/job/82892168619

``` 
Load weight begin. avail mem=60.73 GB
KV Cache is allocated. dtype: torch.bfloat16, #tokens: 76416, KV size: 5.01 GB
Load weight end. elapsed=22.36 s, type=LlamaForCausalLMEagle3, avail mem=11.36 GB, mem usage=6.78 GB.
```

```bash
sglang serve \
    --model-path /root/.cache/modelscope/hub/models/Eco-Tech/Kimi-K2.6-w4a8 \
    --trust-remote-code \
    --attention-backend ascend \
    --device npu \
    --quantization modelslim \
    --dtype bfloat16 \
    --tp-size 32 \
    --nnodes 2 \
    --mem-fraction-static 0.662 \
    --max-running-requests 32 \
    --chunked-prefill-size 262144 \
    --context-length 75000 \
    --enable-multimodal \
    --mm-attention-backend ascend_attn \
    --sampling-backend ascend \
    --enable-dp-attention \
    --dp-size 32 \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --cuda-graph-bs 1 \
    --disable-radix-cache \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path /root/.cache/modelscope/hub/models/lightseekorg/kimi-k2.6-eagle3 \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --speculative-draft-model-quantization unquant \
    --dist-init-addr 172.22.3.166:5000 \
    --node-rank 0 \
    --device npu \
    --host 172.22.3.166 \
    --port 6677
```

``` 
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 32        
Successful requests:                     32        
Benchmark duration (s):                  150.48    
Total input tokens:                      2048000   
Total input text tokens:                 2048000   
Total generated tokens:                  32000     
Total generated tokens (retokenized):    32000     
Request throughput (req/s):              0.21      
Input token throughput (tok/s):          13609.45  
Output token throughput (tok/s):         212.65    
Peak output token throughput (tok/s):    1600.00   
Peak concurrent requests:                32        
Total token throughput (tok/s):          13822.09  
Concurrency:                             31.17     
Accept length:                           3.95      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   146596.95 
Median E2E Latency (ms):                 146405.95 
P90 E2E Latency (ms):                    147210.92 
P95 E2E Latency (ms):                    147921.48 
P99 E2E Latency (ms):                    149673.66 
---------------Time to First Token----------------
Mean TTFT (ms):                          126319.49 
Median TTFT (ms):                        126326.40 
P90 TTFT (ms):                           126339.96 
P95 TTFT (ms):                           126340.62 
P99 TTFT (ms):                           126342.55 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          20.30     
Median TPOT (ms):                        20.09     
P90 TPOT (ms):                           20.90     
P95 TPOT (ms):                           21.61     
P99 TPOT (ms):                           23.37     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           20.30     
Median ITL (ms):                         19.64     
P90 ITL (ms):                            20.26     
P95 ITL (ms):                            20.65     
P99 ITL (ms):                            39.40     
Max ITL (ms):                            86.84     
==================================================

aime25 report table:
┌────────────────┬───────────┬──────────┬──────────┬───────┬─────────┬─────────┐
│ Model          │ Dataset   │ Metric   │ Subset   │   Num │   Score │ Cat.0   │
├────────────────┼───────────┼──────────┼──────────┼───────┼─────────┼─────────┤
│ Kimi-K2.6-w4a8 │ aime25    │ mean_acc │ default  │    30 │  0.9333 │ default │
└────────────────┴───────────┴──────────┴──────────┴───────┴─────────┴─────────┘ 
```


---

kimi27

``` 
Evaluating[aime25]: 100%|██████████████████████████████████████████| 30/30 [22:24<00:00, 44.81s/it]
2026-06-23 09:07:52 - evalscope - INFO: Unified pool finished for aime25.                          
2026-06-23 09:07:52 - evalscope - INFO: Aggregating scores for subset: default                     
2026-06-23 09:07:52 - evalscope - INFO: Generating report...                                       
2026-06-23 09:07:52 - evalscope - INFO:                                                            
aime25 report table:
┌─────────────────────┬───────────┬──────────┬──────────┬───────┬─────────┬─────────┐
│ Model               │ Dataset   │ Metric   │ Subset   │   Num │   Score │ Cat.0   │
├─────────────────────┼───────────┼──────────┼──────────┼───────┼─────────┼─────────┤
│ Kimi-K2.7-Code-w4a8 │ aime25    │ mean_acc │ default  │    30 │       1 │ default │
└─────────────────────┴───────────┴──────────┴──────────┴───────┴─────────┴─────────┘ 
```