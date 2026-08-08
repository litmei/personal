# wget https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl & mv test.jsonl gsm8k.jsonl
# export PYTHONPATH=/home/xjw/code/sglang/python:$PYTHONPATH
python3 -m sglang.test.few_shot_gsm8k \
  --num-questions 200 \
  --num-shots 2 \
  --data-path ./gsm8k.jsonl \
  --max-new-tokens 512 \
  --parallel 200 \
  --host http://0.0.0.0 \
  --port 30000