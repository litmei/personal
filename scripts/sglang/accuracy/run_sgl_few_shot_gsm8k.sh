# 有什么用？
# 对SGLang在线服务进行GSM8K精度跑测

# 如何使用？
# 1. 如果没下载数据集，先手动按照下方链接下载数据集
# 2. 如果环境中没安装sgl，而是使用PYTHON_PATH的配置方式，按照需要进行配置
# 3. 修改详细脚本参数
# 4. 运行该脚本 bash xx.sh

# wget https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
# mv test.jsonl gsm8k.jsonl
# export PYTHONPATH=/home/xjw/code/sglang/common:$PYTHONPATH
python3 -m sglang.test.few_shot_gsm8k \
  --num-questions 200 \
  --num-shots 2 \
  --data-path ./gsm8k.jsonl \
  --max-new-tokens 512 \
  --parallel 200 \
  --host 0.0.0.0 \
  --port 30000