#!/bin/bash
# 评测 tcbench 任务的命令脚本
# 模型: qwen2_vl

export HF_HOME="/root/.cache/huggingface"
export API_TYPE="openai"
export OPENAI_API_URL="YOUR_OPENAI_API_URL"
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
export HF_HOME="/root/.cache/huggingface"
export LMMS_EVAL_USE_CACHE=False
export LMMS_EVAL_HOME="/root/MTC-Bench-Evaluation/lmms-eval"

MODEL_PATH="Qwen/Qwen2-VL-7B-Instruct"

NUM_GPUS=4
    
# 使用 accelerate 启动多 GPU 推理
# 方案 1: 使用 accelerate launch
accelerate launch \
    --multi_gpu \
    --num_processes=$NUM_GPUS \
    -m lmms_eval \
    --model qwen2_vl \
    --model_args pretrained=${MODEL_PATH},use_flash_attention_2=True,max_pixels=602112 \
    --tasks tcbench_video \
    --batch_size 1 \
    --log_samples \
    --limit 10 \
    --output_path ./logs/tcbench_video 2>&1 | tee video.log
#--verbosity DEBUG