# GradingBench Data

**This GitHub repo does not include images or annotation JSON files.**

Download the full dataset from Hugging Face:

**https://huggingface.co/datasets/ERRORSEMI/GradingBench**

## Download

```bash
source scripts/env.sh
bash scripts/download_data.sh
```

Or manually:

```bash
pip install -U huggingface_hub
hf download ERRORSEMI/GradingBench --repo-type dataset --local-dir data
```

## Expected layout after download

```
data/
├── images/
│   ├── L1/{Mathematics,Chinese,English,Science,Humanities}/*.jpg
│   ├── L2/...
│   └── L3/...
├── annotations/
│   ├── L1.json
│   ├── L2.json
│   └── L3.json
├── manifest/
└── schema/
```

## Annotation format

Each `annotations/Lx.json` entry:

```json
{
  "id": "sample_id",
  "subject": "Mathematics",
  "level": "L1",
  "image": "Mathematics/sample_id.jpg",
  "width": 1024,
  "height": 768,
  "marks": [...]
}
```

`image` is relative to `images/Lx/`.

For dataset card metadata used on Hugging Face, see `scripts/hf_dataset_card.md`.
