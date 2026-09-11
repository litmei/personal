#export PYTHONPATH=/home/xjw/code/sglang/python:$PYTHONPATH

######################################
#           USER CONFIG              #
######################################
PLATFORM=npu  # npu, cuda
MODEL_PATH=/home/weights/Kimi-K2.6-w4a8
DRAFT_MODEL_PATH=/home/weights/kimi-k2.6-eagle3
NODE_IPS=(  # 为空表示使用127.0.0.1
  # 1.2.3.4
  # 1.2.3.5
)

# 下面的参数默认会被覆写，一般不用动
MASTER_PORT=8880
MASTER_PORT_INIT=8881
NODE_NUM=1
NODE_ID=0

PLATFORM_LOWER=$(echo "${PLATFORM}" | tr '[:upper:]' '[:lower:]')


######################################
#            MULTI NODES             #
######################################
LOCAL_IPS=$(hostname -I | awk '{print $1" "$2}')

# 方法1：返回当前 LOCAL_IPS 在传入的 IPS 列表（可传 NODE_IPS / NODE_IPS_PREFILL / NODE_IPS_DECODE）中的位置 id
# 找不到时输出 -1
# 用法：get_node_id "${NODE_IPS[@]}"   或   get_node_id "${NODE_IPS_PREFILL}"
get_node_id() {
  local i=0
  local ip
  local local_ip
  for ip in "$@"; do
    for local_ip in ${LOCAL_IPS}; do
      if [ "${local_ip}" = "${ip}" ]; then
        echo "${i}"
        return 0
      fi
    done
    i=$((i + 1))
  done
  echo "-1"
  return 1
}

# if NODE_IPS不为空
if [ ${#NODE_IPS[@]} -gt 0 ]; then
  NODE_NUM=${#NODE_IPS[@]}
  NODE_ID=$(get_node_id "${NODE_IPS[@]}")
  HOST_IP=${NODE_IPS[${NODE_ID}]}
  MASTER_IP=${HOST_IP}
# else 单机模式
else
  MASTER_IP="127.0.0.1"
  NODE_NUM=1
  NODE_ID=0
  HOST_IP=127.0.0.1
fi


######################################
#            ENVS COMMON             #
######################################
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

## [cpu]
#echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
#sysctl -w vm.swappiness=10
#sysctl -w kernel.numa_balancing=0
#sysctl -w kernel.sched_migration_cost_ns=50000
export SGLANG_SET_CPU_AFFINITY=1

## [sgl feat]
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1


######################################
#             ENVS NPU               #
######################################
if [ "${PLATFORM_LOWER}" = "npu" ]; then
  #source /usr/local/Ascend/ascend-toolkit/set_env.sh
  #source /usr/local/Ascend/nnal/atb/set_env.sh

  unset ASCEND_LAUNCH_BLOCKING

  ## [torch]
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
  export STREAMS_PER_DEVICE=32

  ## [hccl]
  export HCCL_CONNECT_TIMEOUT=300
  export HCCL_EXEC_TIMEOUT=300
  export HCCL_BUFFSIZE=400
  export HCCL_OP_EXPANSION_MODE=AIV
  export HCCL_INTRA_PCIE_ENABLE=1
  export HCCL_INTRA_ROCE_ENABLE=0

  ## [net]
  export HCCL_SOCKET_IFNAME=lo
  export GLOO_SOCKET_IFNAME=lo

  ## [cann]
  #export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
  export ACL_DEVICE_SYNC_TIMEOUT=300
  #export TASK_QUEUE_ENABLE=1

  ## [DeepEP] 参数指南 -> https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md
  export DEEPEP_HCCL_BUFFSIZE=2048
  export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=35
  #export SGLANG_DEEPEP_BF16_DISPATCH=1
  #export DEEP_USE_MODE=allgather
  export DEEPEP_HYBRID_DEPLOYMENT=1  # 新Deep混部需要设置此环境变量

  ## [zbal]
  #export HCCL_BUFFSIZE=0
  #unset PYTORCH_NPU_ALLOC_CONF
  #export SGLANG_ZBAL_LOCAL_MEM_SIZE=80000
  #export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
  ## [zbal if use mix alloc]
  #export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
  #export ZBAL_NPU_ALLOC_CONF=use_vmm_for_static_memory:True
  ## [zbal if support graph]
  #export ZBAL_ENABLE_GRAPH=1

  ## [SGL feat]
  #export ASCEND_USE_FIA=1
  #export SGLANG_NPU_USE_MLAPO=1
  export SGLANG_NPU_USE_MULTI_STREAM=1


######################################
#             ENVS CUDA              #
######################################
elif [ "${PLATFORM_LOWER}" = "cuda" ]; then
  ## [net]
  export NCCL_SOCKET_IFNAME=lo
  export GLOO_SOCKET_IFNAME=lo
  ## [cuda]
  #export CUDA_VISIBLE_DEVICES=4,5,6,7
else
  echo "Wrong platform ${PLATFORM}"
  exit 1
fi


######################################
#            SERVER ARGS             #
######################################
ARGS=(
## [BASE]
  --model-path "${MODEL_PATH}"
  --trust-remote-code
  --attention-backend ascend
  --device npu
  --dtype bfloat16
  --mem-fraction-static 0.783
  --max-running-requests 64
  --context-length 16384
  --chunked-prefill-size 4096
  --max-prefill-tokens 8192
  --host "${HOST_IP}"
  --port ${MASTER_PORT}
  --disable-radix-cache
  --nnodes "${NODE_NUM}"
  --node-rank "${NODE_ID}"
  --dist-init-addr "${MASTER_IP}":${MASTER_PORT_INIT}
#  --disable-overlap-schedule

## [PARALLEL]
  --tp-size 4
  --dp-size 4
  --enable-dp-attention
  --moe-a2a-backend deepep
  --deepep-mode auto

## [MM]
#  --enable-multimodal
#  --mm-attention-backend ascend_attn
#  --sampling-backend ascend

## [QUANTIZATION]
  --quantization modelslim

## [GRAPH]
#  --disable-cuda-graph  # 已弃用
  --cuda-graph-backend-prefill disabled
#  --cuda-graph-backend-decode disabled
#  --cuda-graph-bs-prefill 1 2 4 8 16
  --cuda-graph-bs-decode 1 2 4 8 16

## [MTP]
#  --speculative-algorithm EAGLE3
#  --speculative-num-steps 4
#  --speculative-eagle-topk 1
#  --speculative-num-draft-tokens 5
#  --speculative-draft-model-quantization unquant
#  --speculative-draft-model-path "${DRAFT_MODEL_PATH}"

## [OTHER]
  --prefill-delayer-max-delay-passes 200
  --enable-prefill-delayer
  --model-loader-extra-config '{"enable_multithread_load": true}'
)


python3 -m sglang.launch_server "${ARGS[@]}"


exit 0
# shellcheck disable=SC2317
######################################
#           HELP SCRIPTS             #
######################################