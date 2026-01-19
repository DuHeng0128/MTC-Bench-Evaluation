# MTC-Bench Token Compression Evaluation

This repository evaluates **training-free token compression / token pruning** methods (e.g., **FastV**, **VisionZip**, **PruMerge+**) on **MTC-Bench**, using **lmms-eval** as the unified evaluation entry point.

The codebase is adapted from **EffiVLM-Bench** and follows the same high-level integration style (method selection + budget control, with monkeypatch-based injection where applicable).

---

## What’s included

### Supported tasks (lmms-eval)
- `mtcbench_image`
- `mtcbench_video`

### Supported VLMs (example)
- `qwen2_vl` (recommended default)

### Supported token compression methods
- `fastv`
- `visionzip`
- `prumerge+`

> These methods are **training-free** and typically use a **budget** parameter (e.g., `budgets=0.4`) to control the compression ratio.

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
pip install ninja omegaconf flash-attention-softmax-n

conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install nvidia/label/cuda-12.1.1::cuda-nvcc
```

### 2) Set CUDA_HOME (for building flash-attn)

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

# activate.d/cuda_home.sh
cat > $CONDA_PREFIX/etc/conda/activate.d/cuda_home.sh << 'EOF'
#!/bin/bash
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
EOF

# deactivate.d/cuda_home.sh
cat > $CONDA_PREFIX/etc/conda/deactivate.d/cuda_home.sh << 'EOF'
#!/bin/bash
unset CUDA_HOME
EOF
```

### 3) Install flash-attn
```bash
conda activate mllm-efficiency
echo $CUDA_HOME
which nvcc
pip install flash-attn --no-build-isolation
```

### 4) Install lmms-eval (editable)
```bash
cd lmms-eval
pip install -e .
```

### 5) Install qwen2vl (editable) for `qwen2_vl` evaluation
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
  --tasks tcbench_image \
  --batch_size 1 \
  --device cuda:0 \
  --output_path ./results/tcbench_qwen2_vl
```

---

## Option B — Single-command evaluation with token compression

This repository enables compression methods via `--model_args`:
- `method` selects the algorithm (`fastv | visionzip | prumerge+`)
- `budgets` controls the compression budget (0–1)
- some models/methods require extra args such as `use_flash_attention_2=true`

### FastV (example)
```bash
lmms-eval \
  --model qwen2_vl \
  --model_args 'pretrained="Qwen/Qwen2-VL-7B-Instruct",method=fastv,budgets=0.4,use_flash_attention_2=true' \
  --tasks tcbench_image \
  --batch_size 1 \
  --device cuda:0 \
  --output_path ./results/tcbench_qwen2_vl_fastv_b0.4
```

### VisionZip (example)
```bash
lmms-eval \
  --model qwen2_vl \
  --model_args 'pretrained="Qwen/Qwen2-VL-7B-Instruct",method=visionzip,budgets=0.4,use_flash_attention_2=true' \
  --tasks tcbench_image \
  --batch_size 1 \
  --device cuda:0 \
  --output_path ./results/tcbench_qwen2_vl_visionzip_b0.4
```

### PruMerge+ (example)
Because `+` can be interpreted by shells in some contexts, it is safest to quote the value:

```bash
lmms-eval \
  --model qwen2_vl \
  --model_args 'pretrained="Qwen/Qwen2-VL-7B-Instruct",method="prumerge+",budgets=0.4,use_flash_attention_2=true' \
  --tasks tcbench_image \
  --batch_size 1 \
  --device cuda:0 \
  --output_path ./results/tcbench_qwen2_vl_prumergeplus_b0.4
```

To evaluate videos, replace `--tasks tcbench_image` with `--tasks tcbench_video`.

---

## Batch evaluation using `run_example.sh` (recommended)

This repository includes a reference batch script, `run_example.sh`, that:
- activates conda env
- sets required environment variables
- launches `lmms-eval` via `accelerate`
- sweeps over **TASKS × METHODS × BUDGETS**
- writes outputs into a consistent directory structure

The script defines:
- `METHODS=( "prumerge+ prumerge+ use_flash_attention_2=true" "visionzip visionzip use_flash_attention_2=true" "fastv fastv use_flash_attention_2=true" )`
- `BUDGETS=(0.4)`
- `TASKS=("tcbench_image")`
- `MODEL_PATH` and `MODEL_NAME`
- `BASE_COMMAND` using `python3 -m accelerate.commands.launch ... -m lmms_eval ...` fileciteturn6file0

### 1) Review and edit paths
Open `run_example.sh` and update these variables to your local setup (examples shown):

- `ROOT_DIR="/path/to/results"`
- `PYTHONPATH="/path/to/THIS_REPO:/path/to/THIS_REPO/lmms-eval"`
- `CUDA_VISIBLE_DEVICES="0"`
- `MODEL_PATH="/path/to/model_or_hf_cache_or_local_dir"`
- `MODEL_NAME="qwen2_vl"` (or your naming convention)
- `TASKS=("tcbench_image")` or `("tcbench_video")` fileciteturn6file0

### 2) Important security note (API keys)
The example script contains environment variables like `OPENAI_API_KEY`. **Do not commit real keys** into any public repository.
Replace any secrets with placeholders and rely on local `.env`/shell exports instead. fileciteturn6file0

### 3) Run the script
```bash
bash run_example.sh
```

### 4) Result directory naming
The script writes outputs like:

```text
$ROOT_DIR/${MODEL_NAME}_${METHOD}_${TASK}_${BUDGET}_${FILENAME}
```

and skips execution if the output folder already exists. fileciteturn6file0

---

## Outputs

All results are written to the directory specified by `--output_path` (single-command), or `ROOT_DIR` (batch script).

Recommended naming scheme:
- `model_method_task_budget` (e.g., `qwen2_vl_fastv_tcbench_image_0.4_fastv`)
- keep the git commit hash in your experiment log for reproducibility

---

## Suggested repository structure

```text
.
├── lmms-eval/                  # evaluation entry (tasks live here)
│   └── tasks/
│       └── tcbench/            # place the dataset's tcbench folder here
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
