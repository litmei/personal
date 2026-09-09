# modelscope download --dataset gliang1001/ShareGPT_V3_unfiltered_cleaned_split ShareGPT_V3_unfiltered_cleaned_split.json --local_dir ./
# export PYTHONPATH=/home/xjw/code/sglang/python:$PYTHONPATH

ARGS=(
  --backend sglang
  --host 127.0.0.1
  --port 8880
  --model /home/weights/Kimi-K2.5-w4a8
  --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json
  --max-concurrency 24
  --num-prompts 48
  --apply-chat-template

  --dataset-name random
  --random-input 2000
  --random-output 500
  --random-range-ratio 1.0
)

python3 -m sglang.bench_serving "${ARGS[@]}"

