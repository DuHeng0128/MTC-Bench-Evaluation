export HF_HOME="/root/.cache/huggingface"
export API_TYPE="openai"
export OPENAI_API_URL="YOUR_API_URL"
export OPENAI_API_KEY="YOUR_API_KEY"
export LMMS_EVAL_USE_CACHE=True
export LMMS_EVAL_HOME="/root/EffiVLM-Bench/lmms-eval"
export LMMS_EVAL_CACHE_CHUNK_SIZE=256
export LMMS_EVAL_CHUNK_SIZE=1

# 模型路径
MODEL_PATH="/public/huggingface-models/Qwen/Qwen2-VL-7B-Instruct"

# python3 -m lmms_eval \
#     --model qwen2_vl \
#     --model_args pretrained=${MODEL_PATH},use_flash_attention_2=True,max_pixels=602112 \
#     --tasks videomme_long_tcbench \
#     --batch_size 1 \
#     --log_samples \
#     --output_path ./logs/videomme_long_tcbench 2>&1 | tee video.log

# 检测可用的 GPU 数量
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
    
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
    --output_path ./logs/tcbench_video 2>&1 | tee video.log

# 方案 2: 使用 torchrun
# torchrun --nproc_per_node=$NUM_GPUS -m lmms_eval \
#     --model qwen2_vl \
#     --model_args pretrained=${MODEL_PATH},use_flash_attention_2=True \
#     --tasks tcbench_video \
#     --batch_size 1 \
#     --log_samples \
#     --output_path ./logs/tcbench_video 2>&1 | tee video.log
