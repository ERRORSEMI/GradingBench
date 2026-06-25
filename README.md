# GradingBench: Evaluating End-to-End Compositional Reasoning of MLLMs for Automated Exam Grading

**GradingBench** evaluates whether multimodal large language models (MLLMs) can perform **end-to-end automated exam grading** in a single inference pass: given an exam image and a complex grading instruction, the model directly outputs structured JSON for localization, recognition, reasoning, and correctness judgment—without task-specific modules or multi-stage pipelines.

**Dataset (images + annotations) is hosted on Hugging Face**, not in this GitHub repo:

[**GradingBench on Hugging Face**](https://huggingface.co/datasets/ERRORSEMI/GradingBench)

## Benchmark overview

| Item | Description |
|------|-------------|
| **Task levels** | L1 / L2 / L3 (increasing spatial & instruction complexity) |
| **Answer modes** | `answer-free` / `answer-based` (6 settings total) |
| **Scale** | 3,284 question groups across five subjects |

### Dataset dimensions

| Dimension | Values |
|-----------|--------|
| **Subject** | Mathematics, Chinese, English, Science, Humanities |
| **Educational Stage** | Elementary, Junior High, Senior High |
| **Question Type** | Multiple-choice, Fill-in-the-blank, True/False, Problem-Solving |
| **Content Format** | Text-only, Chart-based |

### Task levels

| Level | Input | Setting |
|-------|-------|---------|
| **L1** | Single-question crop | Grade one isolated question |
| **L2** | Full exam page | Grade **specified** question regions on the page |
| **L3** | Full exam page | Locate and grade **all** answer regions on the page |

### Answer modes

| Mode | Description | CLI flag |
|------|-------------|----------|
| **answer-free** | No reference answer in the prompt; the model must reason to obtain the ground-truth answer before grading | `--need_answer False` |
| **answer-based** | Reference answers are provided; the model focuses on answer alignment and correctness judgment | `--need_answer True` |

Together, L1/L2/L3 × answer-free/answer-based form **six standard evaluation settings**.

### Metrics

Reports are written to `results/predictions/Lx/{answer-free\|answer-based}/{model}/1-global_metrics.txt`.

Scoring applies **two filtering gates** on the end-to-end model output:

1. **Text recognition (CER)** — during parse & align, samples whose recognized `text` exceeds the edit-distance threshold are dropped.
2. **Answer verification** — in **answer-free** mode, a text LLM checks semantic equivalence of the model's `answer` to GT; **answer-based** mode skips this step.

| Metric | Meaning |
|--------|---------|
| **End-to-End Accuracy** | Share of question groups where the final grading judgment matches GT |
| **Reasoning Recall** | Share of groups passing answer verification (answer-free only) |
| **FN / FP** | False negatives (right→wrong) and false positives (wrong→right) in final grading |

Metrics are also broken down by **subject**, **educational stage**, **question type**, and **content format**.

See [docs/EVALUATION.md](docs/EVALUATION.md) for the scoring workflow and commands.

## Quick start

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env: MODEL_ROOT, PIGAI_FILTER_MODEL_PATH, optional API keys

source scripts/env.sh
conda activate your-env   # vLLM + transformers env with Qwen-VL support

# 2. Download GradingBench (~467 MB)
bash scripts/download_data.sh
# Or: hf download ERRORSEMI/GradingBench --repo-type dataset --local-dir data

# 3. Install (optional editable install)
pip install -e .

# 4. Run inference + evaluation — e.g. L1 answer-free
export CUDA_VISIBLE_DEVICES=0,1
export TENSOR_PARALLEL_SIZE=2
bash scripts/run_vllm.sh L1 \
  --model Qwen2.5-VL-7B-Instruct --need_answer False

# Metrics only (if predictions already exist):
bash scripts/evaluator.sh L1 --model Qwen2.5-VL-7B-Instruct --need_answer False
```

## Data layout (after download)

```text
data/
├── images/L{1,2,3}/{Mathematics,Chinese,English,Science,Humanities}/*.jpg
├── annotations/L1.json
├── annotations/L2.json
└── annotations/L3.json
```

See [data/README.md](data/README.md) for annotation schema.

## Six evaluation settings

| Setting | Entry |
|---------|-------|
| L1 answer-free | `bash scripts/evaluator.sh L1 --need_answer False --model <M>` |
| L1 answer-based | `bash scripts/evaluator.sh L1 --need_answer True --model <M>` |
| L2 answer-free | `bash scripts/evaluator.sh L2 --need_answer False --model <M>` |
| L2 answer-based | `bash scripts/evaluator.sh L2 --need_answer True --model <M>` |
| L3 answer-free | `bash scripts/evaluator.sh L3 --need_answer False --model <M>` |
| L3 answer-based | `bash scripts/evaluator.sh L3 --need_answer True --model <M>` |

Run all six at once: `bash scripts/evaluate.sh all false --model <model_name>`

## Running models

**Open-source (vLLM):** set `MODEL_ROOT` in `.env`, then use `bash scripts/run_vllm.sh L1|L2|L3`. Configure GPUs via `CUDA_VISIBLE_DEVICES` and `TENSOR_PARALLEL_SIZE`.

**Hosted API (optional):** set `PIGAI_API_APP_ID`, `PIGAI_API_APP_KEY`, and `PIGAI_API_ENDPOINT` in `.env`, then use `bash scripts/run_api.sh L1|L2|L3`.

Predictions are saved under `results/predictions/Lx/{answer-free|answer-based}/{model}/`.

## Repository layout

```text
.
├── README.md
├── pyproject.toml
├── .env.example
├── data/                      # dataset (images + annotations)
├── docs/EVALUATION.md
├── scripts/
│   ├── env.sh                 # environment variables
│   ├── download_data.sh       # download dataset from Hugging Face
│   ├── run_vllm.sh            # L1|L2|L3 inference (vLLM)
│   ├── run_api.sh             # L1|L2|L3 inference (API)
│   ├── evaluator.sh           # Stage1→2→3 metrics
│   └── evaluate.sh            # batch six settings
├── src/gradingbench/          # Python package
│   ├── config/                # level specs, paths settings
│   ├── coords/                # bbox formats & transforms
│   ├── data/                  # annotation I/O
│   ├── eval/                  # metrics & parsing
│   ├── inference/             # vLLM & API runners
│   ├── pipeline/              # stage1/2/3
│   └── prompts/
└── qwentest/                  # local experiment workspace (optional)
```

## Citation

If you use GradingBench in your research, please cite:

```bibtex
@article{gradingbench,
  title   = {GradingBench: Evaluating End-to-End Compositional Reasoning of MLLMs for Automated Exam Grading},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```

## License

Dataset license is specified on the [Hugging Face dataset card](https://huggingface.co/datasets/ERRORSEMI/GradingBench).
