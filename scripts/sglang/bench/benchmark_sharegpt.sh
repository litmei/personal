# modelscope download --dataset gliang1001/ShareGPT_V3_unfiltered_cleaned_split ShareGPT_V3_unfiltered_cleaned_split.json --local_dir ./
# export PYTHONPATH=/home/xjw/code/sglang/common:$PYTHONPATH
python3 -m sglang.bench_serving \
	--dataset-name random \
	--apply-chat-template \
	--random-input 2000 \
	--random-output 500 \
	--random-range-ratio 1.0 \
	--model /home/weights/Kimi-K2.5-w4a8 \
	--backend sglang \
	--max-concurrency 24 \
	--num-prompts 48 \
	--host 127.0.0.1 \
	--port 8880 \
	--dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json
