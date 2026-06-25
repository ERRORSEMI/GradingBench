---
license: other
task_categories:
  - image-to-text
  - visual-question-answering
language:
  - zh
  - en
tags:
  - exam-grading
  - ocr
  - vision-language
  - education
pretty_name: GradingBench
size_categories:
  - 1K<n<10K
---

# GradingBench: Evaluating End-to-End Compositional Reasoning of MLLMs for Automated Exam Grading

Benchmark for evaluating MLLMs on automated exam grading (L1/L2/L3).

Images and annotations are **separated**:

```
data/
├── images/
│   ├── L1/
│   │   ├── Mathematics/   *.jpg
│   │   ├── Chinese/
│   │   ├── English/
│   │   ├── Science/
│   │   └── Humanities/
│   ├── L2/
│   └── L3/
├── annotations/
│   ├── L1.json
│   ├── L2.json
│   └── L3.json
└── manifest/
    └── summary.json
```

## Annotation files

Each `annotations/Lx.json` is a JSON array. Each item:

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

## Code repository

Evaluation code: https://github.com/MaxSuperMax33/GradingBench
