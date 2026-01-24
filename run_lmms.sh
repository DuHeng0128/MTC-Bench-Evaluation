#!/bin/bash

export API_TYPE="openai"
export OPENAI_API_URL="YOUR_API_URL"
export OPENAI_API_KEY="YOUR_API_KEY"
export HF_HOME="/root/.cache/huggingface"
export LMMS_EVAL_USE_CACHE=True
export LMMS_EVAL_CACHE_CHUNK_SIZE=256
export LMMS_EVAL_HOME="/path/to/your/lmms-eval"
export CUDA_VISIBLE_DEVICES="0,1"

MODEL_PATH="/public/huggingface-models/Qwen/Qwen2-VL-7B-Instruct"

python3 -m lmms_eval \
    --model qwen2_vl \
    --model_args pretrained=${MODEL_PATH},use_flash_attention_2=True,device_map="auto" \
    --tasks mtcbench_image \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/mtcbench_image 2>&1 | tee image.log
#--verbosity DEBUG