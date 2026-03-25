#!/bin/bash
# 评测 tcbench 任务的命令脚本
# 模型: qwen2_vl

export HF_HOME="/root/.cache/huggingface"
export API_TYPE="openai"
export OPENAI_API_URL="https://api.nuwaapi.com/v1"
export OPENAI_API_KEY="sk-76iK7BUStTCF2hDqE0l4muxM3hKH1p3s4im9lkurfhn04TdX"
export HF_HOME="/root/.cache/huggingface"
export LMMS_EVAL_USE_CACHE=False
export LMMS_EVAL_HOME="/root/EffiVLM-Bench/lmms-eval"

MODEL_PATH="/public/huggingface-models/Qwen/Qwen2-VL-7B-Instruct"

# python3 -m lmms_eval \
#   --model qwen2_vl \
#   --model_args pretrained=/public/huggingface-models/Qwen/Qwen2-VL-7B-Instruct,use_flash_attention_2=True,max_pixels=602112 \
#   --tasks tcbench_video \
#   --batch_size 1 \
#   --limit 10 \
#   --output_path ./logs/tcbench_video_single

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

# 方案 2: 如果上面的方案不工作，使用 torchrun
# torchrun --nproc_per_node=$NUM_GPUS -m lmms_eval \
#     --model qwen2_vl \
#     --model_args pretrained=${MODEL_PATH},use_flash_attention_2=True \
#     --tasks tcbench_video \
#     --batch_size 1 \
#     --log_samples \
#     --output_path ./logs/tcbench_video 2>&1 | tee video.log

#--verbosity DEBUG