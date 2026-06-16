测试时使用的commit是：282c46133f66d1ae9c2021cba04dc7541526b978

---


今天6月16日，使用今天的commit+一个fix，可以复现1024*1024的性能（需要多尝试几次）：

```bash
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 160       
Successful requests:                     640       
Benchmark duration (s):                  251.31    
Total input tokens:                      28581     
Total input text tokens:                 19621     
Total input vision tokens:               8960      
Total generated tokens:                  655360    
Total generated tokens (retokenized):    654053    
Request throughput (req/s):              2.55      
Input token throughput (tok/s):          113.73    
Output token throughput (tok/s):         2607.78   
Peak output token throughput (tok/s):    5339.00   
Peak concurrent requests:                191       
Total token throughput (tok/s):          2721.51   
Concurrency:                             148.83    
Accept length:                           2.07      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   58442.23  
Median E2E Latency (ms):                 58504.68  
P90 E2E Latency (ms):                    66205.00  
P95 E2E Latency (ms):                    68913.96  
P99 E2E Latency (ms):                    72680.54  
---------------Time to First Token----------------
Mean TTFT (ms):                          7221.74   
Median TTFT (ms):                        5099.79   
P90 TTFT (ms):                           14683.98  
P95 TTFT (ms):                           14734.00  
P99 TTFT (ms):                           17675.47  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          50.07     
Median TPOT (ms):                        50.94     
P90 TPOT (ms):                           56.67     
P95 TPOT (ms):                           57.77     
P99 TPOT (ms):                           60.49     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           50.07     
Median ITL (ms):                         31.19     
P90 ITL (ms):                            89.85     
P95 ITL (ms):                            99.32     
P99 ITL (ms):                            296.86    
Max ITL (ms):                            2735.21   
==================================================
```

另外跑了一个3k5的用例，50ms：

```bash
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 192       
Successful requests:                     768       
Benchmark duration (s):                  337.77    
Total input tokens:                      2688000   
Total input text tokens:                 2688000   
Total generated tokens:                  1152000   
Total generated tokens (retokenized):    1150468   
Request throughput (req/s):              2.27      
Input token throughput (tok/s):          7958.01   
Output token throughput (tok/s):         3410.58   
Peak output token throughput (tok/s):    7016.00   
Peak concurrent requests:                244       
Total token throughput (tok/s):          11368.58  
Concurrency:                             167.20    
Accept length:                           4.61      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   73534.46  
Median E2E Latency (ms):                 70375.86  
P90 E2E Latency (ms):                    92851.40  
P95 E2E Latency (ms):                    105541.17 
P99 E2E Latency (ms):                    135480.72 
---------------Time to First Token----------------
Mean TTFT (ms):                          14673.59  
Median TTFT (ms):                        11495.49  
P90 TTFT (ms):                           25039.65  
P95 TTFT (ms):                           34546.12  
P99 TTFT (ms):                           67999.13  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          39.27     
Median TPOT (ms):                        39.32     
P90 TPOT (ms):                           46.72     
P95 TPOT (ms):                           51.13     
P99 TPOT (ms):                           68.69     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           39.32     
Median ITL (ms):                         25.59     
P90 ITL (ms):                            32.50     
P95 ITL (ms):                            65.57     
P99 ITL (ms):                            268.65    
Max ITL (ms):                            14819.87  
==================================================
```