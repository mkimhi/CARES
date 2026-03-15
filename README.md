# CARES: Context-Aware Resolution Selection for VLMs

Context-Aware Resolution Selection (CARES) is a framework for efficiently selecting appropriate image resolutions when processing images with Vision Language Models (VLMs). Rather than always processing at maximum resolution, CARES learns to identify when sufficient information is present at lower resolutions, reducing computational costs while maintaining model accuracy.

## Overview

CARES implements two complementary approaches:

### 1. Gate-Based Approach
A separate classifier (gate) that predicts whether an image contains sufficient visual information to answer a question at native resolution, or if higher resolution is needed. The gate is trained on top of frozen vision encoders:
- **SigLIP + Text Encoder** (`train_gate_siglip.py`)
- **Multimodal SigLIP** (`train_gate_multimodal.py`)
- **Vision-Language Models** (`train_gate_vlm.py`, `train_gate_vlm2.py`)

### 2. SmolVLM Direct Approach
Trains a lightweight MLP classifier on top of frozen SmolVLM intermediate hidden states to directly predict resolution requirements (`train_smolvlm_gate.py`).

## Project Structure

```
src/
├── gates/                      # Gate-based training scripts
│   ├── train_gate_siglip.py
│   ├── train_gate_multimodal.py
│   ├── train_gate_vlm.py
│   └── train_gate_vlm2.py
├── smolvlm/                    # SmolVLM-based approach
│   └── train_smolvlm_gate.py
├── data_prep/                  # Data generation and preparation
│   ├── gen_training_data.py
│   ├── gen_more_pixels_training_data_qwen.py
│   ├── get_training_dist.py
│   ├── prepare_textvqa.py
│   ├── create_low_res.py
│   └── convert_resolution_quality2parquet.py
├── inference/                  # Inference and evaluation
│   ├── compute_cares_resolutions_gv_data.py
│   └── run_cares_on_gv_data.py
└── utils/                      # Utility functions
    ├── res_stats.py
    ├── res_stats2.py
    ├── confusion.py
    ├── download.py
    ├── valid_res.py
    ├── llave_update.py
    ├── sft_cares.py
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

### Training a Gate Classifier

```bash
# SigLIP-based gate
python src/gates/train_gate_siglip.py \
    --out ./checkpoints/gate_siglip \
    --lr 1e-3 \
    --bsz 128

# SmolVLM-based gate
python src/smolvlm/train_smolvlm_gate.py \
    --parquet data/hardness_data.parquet \
    --model_name HuggingFaceTB/SmolVLM-256M-Instruct \
    --out ./checkpoints/smolvlm_gate
```

### Data Preparation

```bash
# Generate training data
python src/data_prep/gen_training_data.py

# Prepare existing VQA datasets
python src/data_prep/prepare_textvqa.py
```

### Inference

```bash
# Compute resolution predictions on benchmark data
python src/inference/compute_cares_resolutions_gv_data.py
```

## Models

### Base Vision Encoders
- **SigLIP**: google/siglip-so400m-patch14-384
- **CLIP**: openai/clip-vit-large-patch14
- **SmolVLM**: HuggingFaceTB/SmolVLM-256M-Instruct, HuggingFaceTB/SmolVLM-500M-Instruct
- **Qwen**: qwen/qwen-vl

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
