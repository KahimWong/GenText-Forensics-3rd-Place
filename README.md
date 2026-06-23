# SEED

**S**imple ViT and **E**volving Harness for **E**xplainable Text Forgery **D**etection

[![arXiv](https://img.shields.io/badge/arXiv-2606.21138-b31b1b.svg)](https://arxiv.org/pdf/2606.21138)
[![Venue](https://img.shields.io/badge/Venue-ACM_MM_2026-blue)](https://2026.acmmm.org/)
[![Rank](https://img.shields.io/badge/Rank-3rd_🥉-brightgreen)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)](./LICENSE)


---

> 🏆 **3rd Place Solution** for the ACM MM 2026 GenText-Forensics Challenge — detecting, localizing, and explaining text-centric document forgeries.

---

## 🧠 Overview

SEED is a **modular forgery analysis pipeline** with three stages:

| Stage | Component | Description |
|:-----:|-----------|-------------|
| 1️⃣ | **Synthetic Data** | Similarity-guided forgery generation across 5 manipulation types, with paired (clean, forged) sampling |
| 2️⃣ | **ViT Detector** | DINOv3 ViT-L/16 + LoRA adaptation + EoMT mask head — unified detection & localization |
| 3️⃣ | **Meta-Harness** | Evolving MLLM harness that converts detector outputs into structured forensic reports |

<p align="center">
  <img src="fig/seed_overview.png" alt="SEED overview" width="95%">
</p>


## Repository Layout

```text
.
├── base_trainer.py                     # Training utilities & metrics
├── cfg.py                              # Runtime configuration
├── ds.py                               # Datasets & dataloaders
├── main.py                             # Train / validation / inference entry point
├── model/
│   ├── eomt_sep_query.py               # Main detector (DINOv3 + LoRA + EoMT)
│   ├── lora.py                         # Single-expert LoRA modules
│   ├── mask_classification_loss.py     # Mask2Former-style loss
│   └── scale_block.py                  # ConvTranspose upscaling block
├── meta_harness/
│   ├── test_submission.py              # Generate challenge-format reports
│   ├── precompute_submission_artifacts.py
│   ├── harness.py                      # Report generation base class
│   ├── llm_clients.py                  # OpenAI-compatible LLM client
│   ├── overlay.py                      # Mask visualization helpers
│   ├── report_utils.py                 # Report formatting utilities
│   ├── template_report_boxreasons_coordspanrepair.py
│   └── config.yaml                     # LLM API configuration
└── TDOC/                               # Auxiliary training & generation modules
```

## ⚙️ Environment Setup

```bash
# Create a Python 3.10 environment, then install dependencies
pip install -r requirements.txt
```

## 📊 Data Preparation

| Dataset | Description | Link |
|---------|-------------|------|
| RealText-V2 | Original challenge dataset | [vankey/RealText-V2](https://huggingface.co/datasets/vankey/RealText-V2) |
| RealText-V2-Syn25k | Our synthetic data | [Jason37437/RealText-V2-Syn25k](https://huggingface.co/datasets/Jason37437/RealText-V2-Syn25k) |
| Cross-domain test sets | T-SROIE, OSTF, TPIC-13, RTM | [Google Drive](https://drive.google.com/drive/folders/1xn8mELN8etQwRo_PgS5XV6XTKCZasz_A?usp=drive_link) |
| Model Checkpoint | SEED (LoRA rank-1, DINOv3 ViT-L) | [Jason37437/SEED](https://huggingface.co/Jason37437/SEED) / [Google Drive](https://drive.google.com/file/d/1XRbcE2eEdSBdQbyiImg5w9Dn5oMRMKhv/view?usp=drive_link)  |


## 🚀 Training

```bash
# 1. Edit cfg.py → mode='train'
# 2. Set your data paths and GPU count
python main.py
```


## 📈 Evaluation

```bash
# 1. Edit cfg.py → mode='val', eval_mode=['loc','det']
python main.py
```


## 📝 Report Generation

The `meta_harness/` pipeline converts detector outputs → structured Markdown forensic reports.

```bash
# 🔑 Set your OpenAI-compatible API key
export LINKAPI_API_KEY="your-api-key-here"

# 🖼️  Step 1: Precompute overlays, bounding boxes, data URIs
python meta_harness/precompute_submission_artifacts.py

# 📝 Step 2: Generate reports via MLLM
python meta_harness/test_submission.py
```


