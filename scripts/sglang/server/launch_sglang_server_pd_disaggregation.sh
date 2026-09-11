#export PYTHONPATH=/home/xjw/code/sglang/python:$PYTHONPATH

MODEL_PATH=/home/weights/Kimi-K2.6-w4a8
DRAFT_MODEL_PATH=/home/weights/kimi-k2.6-eagle3
NODE_IPS_PREFILL=(
  # 1.2.3.4
  # 1.2.3.5
)
NODE_IPS_DECODE=(
  # 1.2.3.6
  # 1.2.3.7
)
NODE_IPS_ROUTER= # 1.2.3.8

# 下面的参数默认会被覆写，一般不用动，此处留默认值是兼容单机PD分离场景
PLATFORM=npu  # npu, cuda
NODE_CLASS=  # prefill | decode | router
MASTER_IP_PREFILL=127.0.0.1
MASTER_IP_DECODE=127.0.0.1
MASTER_IP_ROUTER=127.0.0.1
MASTER_PORT_PREFILL=8010
MASTER_PORT_DECODE=8110
MASTER_PORT_ROUTER=8880
DISAG_BOOTSTRAP_PORT=8800
# for multinode
MASTER_PORT_INIT_PREFILL=8020
MASTER_PORT_INIT_DECODE=8120
NODE_ID=0
NODE_NUM=1


######################################
#          MULTINODE CONFIG          #
######################################
PLATFORM_LOWER=$(echo "${PLATFORM}" | tr '[:upper:]' '[:lower:]')
LOCAL_IPS=$(hostname -I | awk '{print $1" "$2}')

# 返回当前 LOCAL_IPS 在传入 IP 列表中的位置 id，找不到时输出 -1 并返回非 0
# 用法：get_node_id "${NODE_IPS_PREFILL[@]}"  /  get_node_id "${NODE_IPS_DECODE[@]}"
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

# 节点列表非空时取首位作为 master IP，否则保留默认的 127.0.0.1（单机）
if [ ${#NODE_IPS_PREFILL[@]} -gt 0 ]; then
    MASTER_IP_PREFILL=${NODE_IPS_PREFILL[0]}
fi
if [ ${#NODE_IPS_DECODE[@]} -gt 0 ]; then
    MASTER_IP_DECODE=${NODE_IPS_DECODE[0]}
fi
if [ -n "${NODE_IPS_ROUTER}" ]; then
    MASTER_IP_ROUTER=${NODE_IPS_ROUTER}
fi

# tips: 如果是在单机上拉PD分离，可以把下面这小段（自动判断）注释，手动在几个脚本中配置前面的的 NODE_CLASS，再分别拉起。
# 自动判断本机角色：本机 IP 命中哪个节点列表，就是哪个角色。
NODE_ID=$(get_node_id "${NODE_IPS_PREFILL[@]}")
if [ "${NODE_ID}" -ge 0 ]; then
    NODE_CLASS=prefill
    NODE_NUM=${#NODE_IPS_PREFILL[@]}
else
    NODE_ID=$(get_node_id "${NODE_IPS_DECODE[@]}")
    if [ "${NODE_ID}" -ge 0 ]; then
        NODE_CLASS=decode
        NODE_NUM=${#NODE_IPS_DECODE[@]}
    elif [ -n "${NODE_IPS_ROUTER}" ] && echo " ${LOCAL_IPS} " | grep -q " ${NODE_IPS_ROUTER} "; then
        NODE_CLASS=router
    else
        echo "无法确定本机 NODE_CLASS，请手动在脚本中配置 NODE_CLASS 后分别拉起"
        exit 1
    fi
fi

## [for fast test]
# curl --location "http://${MASTER_IP_ROUTER}:${MASTER_PORT_ROUTER}/generate" \
#   --header 'Content-Type: application/json' \
#   --data '{
#     "text": "介绍下秦始皇",
#     "sampling_params": {
#         "temperature": 0,
#         "max_new_tokens": 20
#     }
#   }'
# exit 0


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


######################################
#             ENVS NPU               #
######################################
if [ "${PLATFORM_LOWER}" = "npu" ]; then
  # source /usr/local/Ascend/ascend-toolkit/set_env.sh
  # source /usr/local/Ascend/nnal/atb/set_env.sh

  unset ASCEND_LAUNCH_BLOCKING

  ## [torch]
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
  export STREAMS_PER_DEVICE=32

  ## [hccl]
  export HCCL_CONNECT_TIMEOUT=300
  export HCCL_EXEC_TIMEOUT=300
  export HCCL_OP_EXPANSION_MODE=AIV
  # export HCCL_INTRA_PCIE_ENABLE=1
  # export HCCL_INTRA_ROCE_ENABLE=0

  ## [net]
  export HCCL_SOCKET_IFNAME=lo
  export GLOO_SOCKET_IFNAME=lo

  ## [cann]
  export ACL_DEVICE_SYNC_TIMEOUT=300

  ## [SGL feat]
  #export ASCEND_USE_FIA=1
  #export SGLANG_NPU_USE_MLAPO=1

  ## [pd disaggregation]
  #export MEMFABRIC_HYBRID_EXTEND_LIB_PATH=
  #export ASCEND_MF_STORE_URL="tcp://${MASTER_IP_PREFILL}:24669"  # 已弃用
  #export MF_CONFIG_STORE_URL="tcp://${MASTER_IP_PREFILL}:24669"
  ## [pd A5]
  #export MF_HYBM_USE_VMM_SEGMENT=1
  #export ASCEND_MF_TRANSFER_PROTOCOL=device_urma

  if [ "${NODE_CLASS}" = "prefill" ]; then
    ## [cann]
    # export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
    export TASK_QUEUE_ENABLE=2
    export HCCL_BUFFSIZE=128
    ## [DeepEP]
    export DEEPEP_HCCL_BUFFSIZE=2048
    # export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
    # export DEEP_USE_ALLTOALL_MODE=1
    ## [zbal]
    # export ZBAL_HCCL_OP=send,recv
    # export ZBAL_NPU_ALLOC_CONF=use_vmm_for_static_memory:True
    ## [SGL feat]
    # export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
    # export SGLANG_PP_LAYER_PARTITION=18,20,24,16
    # export SGLANG_ZBAL_LOCAL_MEM_SIZE=61184
    ## [pd]
    export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=60
    :
  elif [ "${NODE_CLASS}" = "decode" ]; then
    ## [cann]
    # export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
    export TASK_QUEUE_ENABLE=0
    export HCCL_BUFFSIZE=300
    ## [DeepEP] 参数指南 -> https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md
    export DEEPEP_HCCL_BUFFSIZE=2048
    # export DEEP_USE_MODE=allgather
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=40
    # export SGLANG_DEEPEP_BF16_DISPATCH=1
    ## [SGL feat]
    export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
    export SGLANG_ENABLE_SPEC_V2=1
    export SGLANG_NPU_USE_MULTI_STREAM=1
    export SGLANG_SPEC_ENABLE_OVERLAP_REFLOW=1
    :
  else  # router
    :  # router 无需额外环境变量
  fi


######################################
#             ENVS CUDA              #
######################################
elif [ "${PLATFORM_LOWER}" = "cuda" ]; then
  ## [net]
  export NCCL_SOCKET_IFNAME=lo
  export GLOO_SOCKET_IFNAME=lo
  ## [cuda]
  #export CUDA_VISIBLE_DEVICES=4,5,6,7
  :
else
  echo "Wrong platform ${PLATFORM}"
  exit 1
fi


######################################
#       PREFILL SERVER ARGS          #
######################################
ARGS_PREFILL=(
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
  --host "${MASTER_IP_PREFILL}"
  # --host "${NODE_IPS_PREFILL[${NODE_ID}]}"
  --port ${MASTER_PORT_PREFILL}
  --disable-radix-cache
  --nnodes "${NODE_NUM}"
  --node-rank "${NODE_ID}"
  --dist-init-addr "${MASTER_IP_PREFILL}":${MASTER_PORT_INIT_PREFILL}

## [PARALLEL]
  --tp-size 4
  --dp-size 4
  --enable-dp-attention
  --moe-a2a-backend deepep
  --deepep-mode auto

## [pd disaggregation]
 --disaggregation-transfer-backend ascend
## [prefill]
 --disaggregation-mode prefill
 --disaggregation-bootstrap-port ${DISAG_BOOTSTRAP_PORT}
#  --disaggregation-bootstrap-port $((${DISAG_BOOTSTRAP_PORT} + ${NODE_ID}))

## [MM]
#  --enable-multimodal
#  --mm-attention-backend ascend_attn
#  --sampling-backend ascend

## [QUANTIZATION]
  --quantization modelslim

## [GRAPH]
#  --disable-cuda-graph  # 已弃用
  --cuda-graph-backend-prefill disabled
#  --cuda-graph-bs-prefill 1 2 4 8 16

## [OTHER]
  --prefill-delayer-max-delay-passes 200  # 需要dp attn下生效
  --enable-prefill-delayer                # 需要dp attn下生效
  --model-loader-extra-config '{"enable_multithread_load": true}'
)


######################################
#        DECODE SERVER ARGS          #
######################################
ARGS_DECODE=(
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
  --host "${MASTER_IP_DECODE}"
  --port ${MASTER_PORT_DECODE}
  --disable-radix-cache
  --nnodes "${NODE_NUM}"
  --node-rank "${NODE_ID}"
  --dist-init-addr "${MASTER_IP_DECODE}":${MASTER_PORT_INIT_DECODE}

## [PARALLEL]
  --tp-size 4
  --dp-size 4
  --enable-dp-attention
  --moe-a2a-backend deepep
  --deepep-mode auto

## [pd disaggregation]
#  --disaggregation-transfer-backend ascend
## [decode]
#  --disaggregation-mode decode

## [MM]
#  --enable-multimodal
#  --mm-attention-backend ascend_attn
#  --sampling-backend ascend

## [QUANTIZATION]
  --quantization modelslim

## [GRAPH]
#  --disable-cuda-graph  # 已弃用
#  --cuda-graph-backend-decode disabled
  --cuda-graph-bs-decode 1 2 4 8 16

## [MTP]
#  --speculative-algorithm EAGLE3
#  --speculative-num-steps 4
#  --speculative-eagle-topk 1
#  --speculative-num-draft-tokens 5
#  --speculative-draft-model-quantization unquant
#  --speculative-draft-model-path ${DRAFT_MODEL_PATH}

## [OTHER]
  --model-loader-extra-config '{"enable_multithread_load": true}'
)


######################################
#        ROUTER SERVER ARGS          #
######################################
ARGS_ROUTER=(
  --pd-disaggregation
  --policy cache_aware
  --prefill http://"${MASTER_IP_PREFILL}":${MASTER_PORT_PREFILL} ${DISAG_BOOTSTRAP_PORT}
  # --prefill http://"${NODE_IPS_PREFILL[0]}":${MASTER_PORT_PREFILL} $((${DISAG_BOOTSTRAP_PORT} + 0))
  --decode http://"${MASTER_IP_DECODE}":${MASTER_PORT_DECODE}
  --host "${MASTER_IP_ROUTER}"
  --port "${MASTER_PORT_ROUTER}"
  --mini-lb
)


if [ "${NODE_CLASS}" = "prefill" ]; then
  python3 -m sglang.launch_server "${ARGS_PREFILL[@]}"
elif [ "${NODE_CLASS}" = "decode" ]; then
  python3 -m sglang.launch_server "${ARGS_DECODE[@]}"
else  # router
  python3 -m sglang_router.launch_router "${ARGS_ROUTER[@]}"
fi


exit 0
# shellcheck disable=SC2317
######################################
#           HELP SCRIPTS             #
######################################
