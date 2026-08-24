echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=10
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
export SGLANG_SET_CPU_AFFINITY=1

MODEL_PATH=/home/weights/Kimi-K2.6-w4a8
DRAFT_MODEL_PATH=/home/weights/kimi-k2.6-eagle3

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

#source /usr/local/Ascend/ascend-toolkit/set_env.sh
#source /usr/local/Ascend/nnal/atb/set_env.sh

#export HCCL_SOCKET_IFNAME=enp196s0f0
#export GLOO_SOCKET_IFNAME=enp196s0f0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=96
export HCCL_BUFFSIZE=1200
#export HCCL_OP_EXPANSION_MODE=AIV
#export TASK_QUEUE_ENABLE=0

#export PYTHONPATH=/home/xjw/code/sglang/python:$PYTHONPATH


sglang serve \
    --model-path $MODEL_PATH \
    --trust-remote-code \
    --attention-backend ascend \
    --device npu \
    --quantization modelslim \
    --dtype bfloat16 \
    --tp-size 16 \
    --mem-fraction-static 0.783 \
    --max-running-requests 176 \
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
    --cuda-graph-bs 1 2 4 8 11 \
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