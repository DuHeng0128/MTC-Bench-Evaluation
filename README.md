# MTC-Bench Token Compression Evaluation

This repository evaluates **training-free token compression / token pruning** methods (e.g., **FastV**, **VisionZip**, **PruMerge+**) on **MTC-Bench**, using **lmms-eval** as the unified evaluation entry point.

The codebase is adapted from **EffiVLM-Bench** and follows the same high-level integration style (method selection + budget control, with monkeypatch-based injection where applicable).

---

## What’s included

### Supported tasks (lmms-eval)
- `mtcbench_image`
- `mtcbench_video`

### Supported VLMs
- `Qwen2-VL`
- `Qwen2.5-VL`
- `LLaVA-OneVision`
- `LLaVA-OneVision 1.5`
- `InternVL2.5`
- `InternVL3`

### Supported token compression methods
- `fastv`
- `visionzip`
- `prumerge+`
- `dart`
- `pdrop`

> These methods are **training-free** and use a **`budgets`** parameter (e.g., `budgets=0.4`) to control the keep ratio of visual tokens.

---

## Dataset and task registration

### 1) Download MTC-Bench
Dataset on HuggingFace:

- `https://huggingface.co/datasets/DuHeng0128/MTC-Bench`

Download the dataset with your preferred method (e.g., `huggingface-cli`, scripts, or manual download).

### 2) Extract and Configure the Dataset
Navigate to your downloaded MTC-Bench folder and run the following commands:

1. Extract the dataset: 
   ```bash
   cd /path/to/your/MTC-Bench
   cat MTC-Bench.tar.* | tar -xvf -
   ```
   This creates the mtcbench data folder.

2. Update YAML configuration paths:

   Run the script `update_yaml_paths.py` (provided in the Hugging Face repo).

   Important: Open `update_yaml_paths.py` and set the `YOUR_LOCAL_DATASET_PATH` variable to the absolute path of the extracted `mtcbench` folder.

   ```python
   # Example Path
   YOUR_LOCAL_DATASET_PATH = '/root/data/MTC-Bench'
   ```

3. Run the script:

   ```bash
   python update_yaml_paths.py
   ```
   This updates all task YAML files to use your local dataset path.

### 3) Register tasks in lmms-eval
Move the dataset folder `mtcbench` into:

```bash
lmms-eval/tasks/mtcbench
```

After that, lmms-eval can discover:
- `mtcbench_image`
- `mtcbench_video`

---

## Installation
Below is a practical installation flow (CUDA / PyTorch / flash-attn / lmms-eval / qwen2vl).

### 1) Create an environment and install dependencies

```bash
conda create -n mllm-efficiency python=3.10
conda activate mllm-efficiency

sudo apt-get update
sudo apt-get install -y openjdk-11-jre-headless

pip install -r requirements.txt
pip install ninja flash-attention-softmax-n omegaconf==2.0.0
```

### 2) Install flash-attn
```bash
pip install flash-attn --no-build-isolation
```

### 3) Install lmms-eval (editable)
```bash
cd lmms-eval
pip install -e .
```

### 4) Install qwen2vl (editable) for `qwen2_vl` evaluation
```bash
cd ../qwen2vl
pip install -e .
pip install qwen-vl-utils
```

---

## Running evaluations

You can run evaluations in two ways:

1. **Single command** (quick manual invocation)
2. **Batch script** (`run_example.sh`) to sweep tasks/methods/budgets via `accelerate`

The batch script approach is recommended for controlled experiments and producing consistent result folder structures.

---

## Option A — Single-command evaluation (baseline)

```bash
lmms-eval \
  --model qwen2_vl \
  --model_args pretrained="Qwen/Qwen2-VL-7B-Instruct" \
  --tasks mtcbench_image \
  --batch_size 1 \
  --device cuda:0 \
  --output_path ./results/mtcbench_qwen2_vl
```

---

## Option B — Single-command evaluation with token compression

This repository enables compression methods via `--model_args`:
- `method` selects the algorithm (`fastv | visionzip | prumerge+ | pdrop`)
- `budgets` controls the compression budget (0–1)
- some models/methods require extra args such as `use_flash_attention_2=true`

### FastV (example)
```bash
lmms-eval \
  --model qwen2_vl \
  --model_args 'pretrained="Qwen/Qwen2-VL-7B-Instruct",method=fastv,budgets=0.4,use_flash_attention_2=true' \
  --tasks mtcbench_image \
  --batch_size 1 \
  --device cuda:0 \
  --output_path ./results/mtcbench_qwen2_vl_fastv_b0.4
```

### VisionZip (example)
```bash
lmms-eval \
  --model qwen2_vl \
  --model_args 'pretrained="Qwen/Qwen2-VL-7B-Instruct",method=visionzip,budgets=0.4,use_flash_attention_2=true' \
  --tasks mtcbench_image \
  --batch_size 1 \
  --device cuda:0 \
  --output_path ./results/mtcbench_qwen2_vl_visionzip_b0.4
```

### DART (example)
```bash
lmms-eval \
  --model qwen2_vl \
  --model_args 'pretrained="Qwen/Qwen2-VL-7B-Instruct",method=dart,budgets=0.4,use_flash_attention_2=true' \
  --tasks mtcbench_image \
  --batch_size 1 \
  --device cuda:0 \
  --output_path ./results/mtcbench_qwen2_vl_dart_b0.4
```

### PruMerge+ (example)
Because `+` can be interpreted by shells in some contexts, it is safest to quote the value:

```bash
lmms-eval \
  --model qwen2_vl \
  --model_args 'pretrained="Qwen/Qwen2-VL-7B-Instruct",method="prumerge+",budgets=0.4,use_flash_attention_2=true' \
  --tasks mtcbench_image \
  --batch_size 1 \
  --device cuda:0 \
  --output_path ./results/mtcbench_qwen2_vl_prumergeplus_b0.4
```

### PDrop (example)
`pdrop` currently provides staged visual-token pruning examples for `Qwen2-VL` and `LLaVA-OneVision 1.5`.
It requires two extra arguments:

- `layer_list`: pruning layers
- `image_token_ratio_list`: retained image-token ratios after each stage

The two lists must have the same length, and `image_token_ratio_list` should be non-increasing across stages.

```bash
lmms-eval \
  --model qwen2_vl_with_kvcache \
  --model_args 'pretrained="Qwen/Qwen2-VL-7B-Instruct",method=pdrop,budgets=0.4,layer_list="[7,14,21]",image_token_ratio_list="[0.3893,0.1516,0.0590]",use_flash_attention_2=true' \
  --tasks mtcbench_image \
  --batch_size 1 \
  --device cuda:0 \
  --output_path ./results/mtcbench_qwen2_vl_pdrop_b0.4
```

To evaluate videos, replace `--tasks mtcbench_image` with `--tasks mtcbench_video`.

---

## Batch evaluation using `run_example.sh` (recommended)

`run_example.sh` launches `lmms-eval` via `accelerate` and sweeps **TASKS × METHODS × BUDGETS** automatically, skipping any output folder that already exists.

### Key variables to configure

Open `run_example.sh` and set the following:

| Variable | Description | Required |
|---|---|---|
| `ROOT_DIR` | Directory where results are written | ✅ |
| `MODEL_PATH` | Local path or HuggingFace model ID | ✅ |
| `MODEL_NAME` | Label used in the output folder name | ✅ |
| `NUM_PROCESSES` | Number of GPUs (default: `4`) | optional |
| `CUDA_VISIBLE_DEVICES` | GPU indices to use (default: `"0,1,2,3"`) | optional |
| `TASKS` | List of tasks, e.g. `("mtcbench_image")` or `("mtcbench_video")` | optional |
| `BUDGETS` | Keep-ratio list, e.g. `(0.4)` or `(0.25 0.5)` | optional |
| `OPENAI_API_KEY` / `OPENAI_API_URL` | Required only for tasks with GPT-based evaluation | optional |
| `--log_samples` | Save per-sample model outputs to the output directory | optional |
| `--reuse_responses` | Load cached model responses from a previous run to skip re-inference | optional |

> **Security note**: Do not commit real API keys. Set them via local shell exports or a `.env` file instead.

### Selecting a model

Set `--model` in `BASE_COMMAND` to match your target VLM:

```bash
--model qwen2_vl_with_kvcache            # Qwen2-VL
--model qwen2_5_vl_with_kvcache          # Qwen2.5-VL
--model qwen3_vl_with_kvcache            # Qwen3-VL
--model llava_onevision_with_kvcache     # LLaVA-OneVision
--model llava_onevision1_5_with_kvcache  # LLaVA-OneVision 1.5
--model internvl2_with_kvcache           # InternVL2.5
--model internvl3_with_kvcache           # InternVL3
```

### Selecting methods

Each entry in `METHODS` has three space-separated fields:

```
"<method_name>  <output_label>  [additional_model_args]"
```

The third field is **optional** and is appended to `--model_args` verbatim. Examples:

```bash
METHODS=(
    # Token pruning — Qwen2-VL / Qwen2.5-VL (requires flash-attention)
    "fastv     fastv     use_flash_attention_2=true"
    "pdrop     pdrop     layer_list='[7,14,21]',image_token_ratio_list='[0.3893,0.1516,0.0590]',use_flash_attention_2=true"
    "visionzip visionzip use_flash_attention_2=true"
    "prumerge+ prumerge+ use_flash_attention_2=true"
    "dart      dart      use_flash_attention_2=true"
    # "h2o        head      head_adaptive=True,use_flash_attention_2=true"
    # "snapkv     head      head_adaptive=True,pooling=avgpool,use_flash_attention_2=true"
    # "pyramidkv  head      head_adaptive=True,pooling=avgpool,use_flash_attention_2=true"
    # "look-m     merge     merge=True,use_flash_attention_2=true"
    # "streamingllm streamingllm use_flash_attention_2=true"

    # Token pruning — InternVL2.5-38B (multi-GPU model parallel)
    # "fastv     fastv     device_map=auto"
    # "visionzip visionzip device_map=auto"
    # "prumerge+ prumerge+ device_map=auto"

    # KV cache methods — LLaVA-OneVision
    # "fastv     fastv"
    # "visionzip visionzip"
    # "prumerge+ prumerge+"
)
```

### Run the script

```bash
bash run_example.sh
```

### Result directory naming

Outputs are written to:

```text
$ROOT_DIR/${MODEL_NAME}_${METHOD}_${TASK}_${BUDGET}_${FILENAME}
```

If the folder already exists, that combination is skipped automatically.

---


## Outputs

All results are written to the directory specified by `--output_path` (single-command), or `ROOT_DIR` (batch script).

Recommended naming scheme:
- `model_method_task_budget` (e.g., `qwen2_vl_fastv_mtcbench_image_0.4_fastv`)
- keep the git commit hash in your experiment log for reproducibility

---

## Suggested repository structure

```text
.
├── lmms-eval/                  # evaluation entry (tasks live here)
│   └── tasks/
│       └── mtcbench/            # place the dataset's mtcbench folder here
├── qwen2vl/                    # qwen2-vl support (editable install)
├── kv_cache_compression/       # optional: if you extend KV cache compression
├── token_compression/          # optional: keep method patches in one place
├── run_example.sh              # batch runner for tasks/methods/budgets
└── README.md
```

---

## Acknowledgements

This repository is adapted from **EffiVLM-Bench** and reuses its general integration approach for training-free acceleration methods. We also acknowledge the original open-source implementations and papers behind FastV, VisionZip, and PruMerge(+).

---

## Citation
If you find this benchmark helpful or use it in your research, please consider citing our survey paper:

```bibtex
@article{yao2026towards,
  title={Towards Efficient Multimodal Large Language Models: A Survey on Token Compression},
  author={Yao, Linli and Xing, Long and Shi, Yang and Li, Sida and Liu, Yuanxin and Dong, Yuhao and Zhang, Yi-Fan and Li, Lei and Dong, Qingxiu and Dong, Xiaoyi and others},
  journal={Authorea Preprints},
  year={2026},
  publisher={Authorea}
}
```
