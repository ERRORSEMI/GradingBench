# Evaluation Guide

## End-to-end inference

Each sample is evaluated with **one MLLM forward pass**: the model receives an exam image plus a grading instruction and returns a structured JSON response. No task-specific modules or cascaded inference stages are used at test time.

## Tasks

| Level | Input | Setting |
|-------|-------|---------|
| **L1** | Single-question crop | Grade one isolated question |
| **L2** | Full page | Grade **specified** question regions |
| **L3** | Full page | Locate and grade **all** answer regions |

## Answer modes

| Mode | CLI flag | Output dir |
|------|----------|------------|
| **answer-free** | `--need_answer False` | `results/predictions/Lx/answer-free/{model}/` |
| **answer-based** | `--need_answer True` | `results/predictions/Lx/answer-based/{model}/` |

In **answer-free** mode, reference answers are withheld from the prompt; the model must derive them before grading.

In **answer-based** mode, reference answers are provided in the prompt as the grading basis.

## Scoring workflow

After end-to-end inference, the evaluation scripts parse and score model outputs:

```text
End-to-end inference (MLLM)
    → per-sample *_raw.json
Parse & align
    → 0-global_results_raw.json
Answer verification (Qwen3-8B, answer-free only)
    → 0-global_results_filtered.json
Metric aggregation
    → 1-global_metrics.txt
```

### Parse & align

- Parse each `*_raw.json` into structured predictions.
- Match predictions to GT answer boxes (IoU-first, then nearest-center fallback).
- **Filter by text recognition (CER):** if the normalized edit distance between predicted `text` and GT handwriting exceeds 0.7, the sample is dropped and does not enter downstream scoring.

### Answer verification

- In **answer-free** mode, a text-only LLM (`PIGAI_FILTER_MODEL_PATH`, default Qwen3-8B) checks whether the model's `answer` is semantically equivalent to GT. Samples that fail are dropped before computing final accuracy.
- In **answer-based** mode, this step is skipped.

### Metrics

| Metric | Definition |
|--------|------------|
| **End-to-End Accuracy** | `# correct grading judgments / # GT question groups` |
| **Reasoning Recall** | `# answer-verified groups / # GT question groups` (answer-free; all groups kept in answer-based) |
| **FN / FP** | Grading errors: right→wrong (FN), wrong→right (FP) |

Breakdowns are reported for **subject**, **educational stage**, **question type**, and **content format**.

## Shell templates (per level)

| Script | Purpose |
|--------|---------|
| `run_vllm.sh` | Open-source vLLM inference + optional evaluation |
| `run_api.sh` | Hosted VL API inference + optional evaluation |
| `evaluator.sh` | Scoring only (parse → verify → metrics) |

## Dataset dimensions

| Dimension | Values |
|-----------|--------|
| **Subject** | Mathematics, Chinese, English, Science, Humanities |
| **Educational Stage** | Elementary, Junior High, Senior High |
| **Question Type** | Multiple-choice, Fill-in-the-blank, True/False, Problem-Solving |
| **Content Format** | Text-only, Chart-based |

## Commands

```bash
source scripts/env.sh

# Open-source (set GPU via env)
export CUDA_VISIBLE_DEVICES=0
export TENSOR_PARALLEL_SIZE=1
bash workpy/L1-pigaipy/shells/run_vllm.sh \
  --model Qwen2.5-VL-7B-Instruct --need_answer False

# API model
bash workpy/L1-pigaipy/shells/run_api.sh --model glm-4.6v --need_answer False

# Evaluate existing predictions
bash workpy/L1-pigaipy/shells/evaluator.sh --model glm-4.6v --need_answer False
bash scripts/evaluate.sh all false --model <model_name>
```

## Outputs

```text
results/predictions/L1/answer-free/{model}/
├── *_raw.json                         # end-to-end MLLM responses
├── 0-global_results_raw.json          # parsed & aligned results
├── 0-global_results_filtered.json     # answer-verified results
└── 1-global_metrics.txt               # final metrics
```
