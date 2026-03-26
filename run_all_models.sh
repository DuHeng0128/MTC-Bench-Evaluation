source ~/.bashrc
source /root/miniconda3/bin/activate
conda activate mtcbench-eval

ROOT_DIR="/root/MTC-Bench-Evaluation/results"
NUM_PROCESSES=4
TASKS=("mtcbench_image")
BUDGETS=(0.4)

export HF_ENDPOINT="https://hf-mirror.com"
export CONDA_DEFAULT_ENV="mtcbench-eval"
export PATH="/root/miniconda3/envs/mtcbench-eval/bin:$PATH"
export OPENAI_API_URL="YOUR_OPENAI_API_URL"
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

LAUNCH="python3 -m accelerate.commands.launch \
    --main_process_port=28176 \
    --mixed_precision=bf16 \
    --num_processes=$NUM_PROCESSES \
    -m lmms_eval \
    --batch_size 1 \
    --log_samples \
    --reuse_responses"


run_eval() {
    local MODEL_CLASS="$1"
    local MODEL_NAME="$2"
    local MODEL_PATH="$3"
    local -n METHODS="$4"

    for TASK in "${TASKS[@]}"; do
        for METHOD_CONFIG in "${METHODS[@]}"; do
            local METHOD FILENAME ADDITIONAL_ARGS
            METHOD=$(echo "$METHOD_CONFIG"   | awk '{print $1}')
            FILENAME=$(echo "$METHOD_CONFIG" | awk '{print $2}')
            ADDITIONAL_ARGS=$(echo "$METHOD_CONFIG" | awk '{$1="";$2=""; print $0}' | xargs)

            for BUDGET in "${BUDGETS[@]}"; do
                local OUTPUT_PATH="$ROOT_DIR/${MODEL_NAME}_${METHOD}_${TASK}_${BUDGET}_${FILENAME}"

                if find "$OUTPUT_PATH" -name "*_results.json" -type f 2>/dev/null | grep -q .; then
                    echo "[SKIP] Results already exist: $OUTPUT_PATH"
                    continue
                fi

                mkdir -p "$OUTPUT_PATH"
                local LOG_FILE="$OUTPUT_PATH/eval.log"

                local MODEL_ARGS="pretrained=${MODEL_PATH},method=${METHOD},budgets=${BUDGET}"
                [ -n "$ADDITIONAL_ARGS" ] && MODEL_ARGS="${MODEL_ARGS},${ADDITIONAL_ARGS}"

                local COMMAND="${LAUNCH} \
                    --model ${MODEL_CLASS} \
                    --tasks ${TASK} \
                    --output_path ${OUTPUT_PATH} \
                    --log_samples_suffix ${TASK} \
                    --model_args \"${MODEL_ARGS}\""

                echo "========================================================"
                echo "[START] $(date '+%Y-%m-%d %H:%M:%S')"
                echo "  Model : $MODEL_NAME ($MODEL_CLASS)"
                echo "  Method: $METHOD  Budget: $BUDGET"
                echo "  Output: $OUTPUT_PATH"
                echo "  Log   : $LOG_FILE"
                echo "========================================================"

                # Redirect both stdout and stderr to log; also tee to terminal
                eval "${COMMAND}" 2>&1 | tee "$LOG_FILE"

            done
        done
    done
}

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Qwen2-VL-7B
# ═══════════════════════════════════════════════════════════════════════════════
METHODS_QWEN2VL=(
    "fastv     fastv     use_flash_attention_2=true"
    "visionzip visionzip use_flash_attention_2=true"
    "prumerge+ prumerge+ use_flash_attention_2=true"
    # "dart      dart      use_flash_attention_2=true"
    # "h2o        head      head_adaptive=True,use_flash_attention_2=true"
    # "snapkv     head      head_adaptive=True,pooling=avgpool,use_flash_attention_2=true"
    # "pyramidkv  head      head_adaptive=True,pooling=avgpool,use_flash_attention_2=true"
    # "look-m     merge     merge=True,use_flash_attention_2=true"
    # "streamingllm streamingllm use_flash_attention_2=true"
)
run_eval "qwen2_vl_with_kvcache" \
         "qwen2-vl" \
         "Qwen/Qwen2-VL-7B-Instruct" \
         METHODS_QWEN2VL

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Qwen2.5-VL-7B
# ═══════════════════════════════════════════════════════════════════════════════
METHODS_QWEN25VL=(
    "fastv     fastv     use_flash_attention_2=true"
    "visionzip visionzip use_flash_attention_2=true"
    "prumerge+ prumerge+ use_flash_attention_2=true"
    # "dart      dart      use_flash_attention_2=true"
    # "h2o        head      head_adaptive=True,use_flash_attention_2=true"
    # "snapkv     head      head_adaptive=True,pooling=avgpool,use_flash_attention_2=true"
    # "pyramidkv  head      head_adaptive=True,pooling=avgpool,use_flash_attention_2=true"
    # "look-m     merge     merge=True,use_flash_attention_2=true"
    # "streamingllm streamingllm use_flash_attention_2=true"
)
run_eval "qwen2_5_vl_with_kvcache" \
         "qwen2_5-vl" \
         "Qwen/Qwen2.5-VL-7B-Instruct" \
         METHODS_QWEN25VL

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Qwen3-VL-8B
# ═══════════════════════════════════════════════════════════════════════════════
# METHODS_QWEN3VL=(
#     "fastv     fastv     use_flash_attention_2=true"
#     "visionzip visionzip use_flash_attention_2=true"
#     "prumerge+ prumerge+ use_flash_attention_2=true"
#     # "h2o        head      head_adaptive=True,use_flash_attention_2=true"
#     # "snapkv     head      head_adaptive=True,pooling=avgpool,use_flash_attention_2=true"
#     # "pyramidkv  head      head_adaptive=True,pooling=avgpool,use_flash_attention_2=true"
#     # "look-m     merge     merge=True,use_flash_attention_2=true"
#     # "streamingllm streamingllm use_flash_attention_2=true"
# )
# run_eval "qwen3_vl_with_kvcache" \
#          "qwen3-vl" \
#          "Qwen/Qwen3-VL-8B-Instruct" \
#          METHODS_QWEN3VL

# ═══════════════════════════════════════════════════════════════════════════════
# 4. LLaVA-OneVision-7B
# ═══════════════════════════════════════════════════════════════════════════════
METHODS_LLAVA_OV=(
    "fastv     fastv"
    "visionzip visionzip"
    "prumerge+ prumerge+"
    # "dart      dart"
    # "h2o        head      head_adaptive=True"
    # "snapkv     head      head_adaptive=True,pooling=avgpool"
    # "pyramidkv  head      head_adaptive=True,pooling=avgpool"
    # "look-m     merge     merge=True"
    # "streamingllm streamingllm"
)
run_eval "llava_onevision_with_kvcache" \
         "llava-onevision" \
         "/root/data2/shared/models/llava-onevision-qwen2-7b-ov" \
         METHODS_LLAVA_OV

# ═══════════════════════════════════════════════════════════════════════════════
# 5. LLaVA-OneVision-1.5-8B
# ═══════════════════════════════════════════════════════════════════════════════
METHODS_LLAVA_OV15=(
    "fastv     fastv"
    "visionzip visionzip"
    "prumerge+ prumerge+"
    # "dart      dart"
    # "h2o        head      head_adaptive=True"
    # "snapkv     head      head_adaptive=True,pooling=avgpool"
    # "pyramidkv  head      head_adaptive=True,pooling=avgpool"
    # "look-m     merge     merge=True"
    # "streamingllm streamingllm"
)
run_eval "llava_onevision1_5_with_kvcache" \
         "llava-onevision-1.5" \
         "/root/data2/shared/models/LLaVA-OneVision-1.5-8B-Instruct" \
         METHODS_LLAVA_OV15

# ═══════════════════════════════════════════════════════════════════════════════
# 6. InternVL3-8B
# ═══════════════════════════════════════════════════════════════════════════════
METHODS_INTERNVL3=(
    "fastv     fastv     device_map=auto"
    "visionzip visionzip device_map=auto"
    "prumerge+ prumerge+ device_map=auto"
    # "dart      dart      device_map=auto"
    # "h2o        head      head_adaptive=True,device_map=auto"
    # "snapkv     head      head_adaptive=True,pooling=avgpool,device_map=auto"
    # "pyramidkv  head      head_adaptive=True,pooling=avgpool,device_map=auto"
    # "look-m     merge     merge=True,device_map=auto"
    # "streamingllm streamingllm device_map=auto"
)
run_eval "internvl3_with_kvcache" \
         "internvl3" \
         "OpenGVLab/InternVL3-8B" \
         METHODS_INTERNVL3

echo "========================================================"
echo "All evaluations completed: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
