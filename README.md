# CARES: Context-Aware Resolution Selection for VLMs

Context-Aware Resolution Selection (CARES) is a framework for efficiently selecting appropriate image resolutions when processing images with Vision Language Models (VLMs). Rather than always processing at maximum resolution, CARES learns to identify when sufficient information is present at lower resolutions, reducing computational costs while maintaining model accuracy.

## Overview

CARES provides two complementary approaches:

### 1. SmolVLM Classifier
External classifier trained on top of frozen SmolVLM intermediate hidden states:
- **Model**: HuggingFaceTB/SmolVLM-256M/500M-Instruct
- **Approach**: MLP classifier on frozen features
- **Training**: Efficient with minimal parameter updates
- **Output**: Resolution class prediction (multiclass or binary)
- **Best for**: Resource-constrained environments, quick inference

### 2. Granite-Docling Autoregressive
Fine-tuned autoregressive model that directly learns resolution sufficiency:
- **Model**: IBM Granite-Docling
- **Approach**: SFT (Supervised Fine-Tuning) with LoRA adapters
- **Training**: End-to-end model fine-tuning
- **Output**: Direct autoregressive resolution prediction
- **Best for**: Production deployments, easy hosting, interpretable outputs

## Project Structure

```
src/
├── training/
│   ├── train_smolvlm_gate.py   # SmolVLM classifier on frozen features
│   └── train_granite_sft.py    # Granite-Docling SFT with LoRA
└── utils/                      # Utility functions
    ├── res_stats2.py           # Resolution statistics
    ├── confusion.py
    ├── download.py
    ├── upload_lora.py
    ├── valid_res.py
    └── llave_update.py
```

## Installation

```bash
# Clone the repository
git clone https://github.com/mkimhi/CARES.git
cd CARES

# Install dependencies
pip install transformers torch accelerate datasets sklearn safetensors peft trl pandas pillow
```

## Usage

### Approach 1: SmolVLM Classifier

```bash
# Basic training with 256M model
python src/training/train_smolvlm_gate.py \
    --parquet data/hardness_data.parquet \
    --model_name HuggingFaceTB/SmolVLM-256M-Instruct \
    --out ./checkpoints/smolvlm_gate \
    --bsz 64 \
    --epochs 10 \
    --lr 1e-4

# With 500M model and binary classification
python src/training/train_smolvlm_gate.py \
    --parquet data/hardness_data.parquet \
    --model_name HuggingFaceTB/SmolVLM-500M-Instruct \
    --out ./checkpoints/smolvlm_500m \
    --binary \
    --bsz 32
```

### Approach 2: Granite-Docling Autoregressive

```bash
# Train with SFT and LoRA adapters
python src/training/train_granite_sft.py \
    --parquet data/hardness_data.parquet \
    --output_dir ./checkpoints/granite_sft \
    --batch_size 32 \
    --num_epochs 3 \
    --use_lora
```

### Analysis and Utilities

```bash
# Analyze training data distribution
python src/utils/res_stats2.py --input data/training.parquet

# Generate confusion matrix
python src/utils/confusion.py --file1 predictions1.json --file2 predictions2.json
```

## Models

### SmolVLM Classifier
- **SmolVLM-256M-Instruct**: Lightweight model, good for on-device deployment
- **SmolVLM-500M-Instruct**: Larger variant with improved accuracy

### Granite-Docling Autoregressive
- **IBM Granite-Docling**: Foundation model for document understanding
- **LoRA Adapters**: Efficient parameter-efficient fine-tuning

### Datasets Used
- TextVQA
- DocVQA
- ChartQA
- InfographicVQA
- HME100K

## Performance

CARES achieves significant computational savings by:
- Reducing average resolution processing by up to 80% 
- Maintaining question answering accuracy within 1% of full-resolution baselines
- Supporting multi-resolution inference strategies

## Citation

```
@misc{kimhi2025carescontextawareresolutionselector,
      title={CARES: Context-Aware Resolution Selector for VLMs}, 
      author={Moshe Kimhi and Nimrod Shabtay and Raja Giryes and Chaim Baskin and Eli Schwartz},
      year={2025},
      eprint={2510.19496},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
}
```
