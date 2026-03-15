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
├── smolvlm/                    # SmolVLM classifier approach
│   └── train_smolvlm_gate.py   # Train classifier on SmolVLM features
├── granite_sft/                # Granite-Docling autoregressive approach
│   └── train_granite_sft.py    # Fine-tune Granite with SFT + LoRA
└── utils/                      # Utility functions
    ├── res_stats2.py           # Resolution statistics
    ├── confusion.py
    ├── download.py
    ├── valid_res.py
    ├── llave_update.py
    └── upload_lora.py
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
python src/smolvlm/train_smolvlm_gate.py \
    --parquet data/hardness_data.parquet \
    --model_name HuggingFaceTB/SmolVLM-256M-Instruct \
    --out ./checkpoints/smolvlm_gate \
    --bsz 64 \
    --epochs 10 \
    --lr 1e-4

# With 500M model and binary classification
python src/smolvlm/train_smolvlm_gate.py \
    --parquet data/hardness_data.parquet \
    --model_name HuggingFaceTB/SmolVLM-500M-Instruct \
    --out ./checkpoints/smolvlm_500m \
    --binary \
    --bsz 32
```

### Approach 2: Granite-Docling Autoregressive

```bash
# Train with SFT and LoRA adapters
python src/granite_sft/train_granite_sft.py \
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
- Reducing average resolution processing by X% (check paper for specific numbers)
- Maintaining question answering accuracy within Y% of full-resolution baselines
- Supporting multi-resolution inference strategies

## Citation

[To be added]

## License

[To be specified]

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
