# CARES: Context-Aware Resolution Selection for VLMs

Context-Aware Resolution Selection (CARES) is a framework for efficiently selecting appropriate image resolutions when processing images with Vision Language Models (VLMs). Rather than always processing at maximum resolution, CARES learns to identify when sufficient information is present at lower resolutions, reducing computational costs while maintaining model accuracy.

## Overview

CARES implements a lightweight direct prediction approach using SmolVLM:

### SmolVLM Direct Approach
Trains a lightweight MLP classifier on top of frozen SmolVLM intermediate hidden states to directly predict resolution requirements. This approach:
- Works with HuggingFaceTB/SmolVLM-256M-Instruct and SmolVLM-500M-Instruct
- Freezes all model weights, learning only a classification head
- Supports multiclass (3-class: low/medium/high) and binary classification modes
- Achieves efficient on-device inference with minimal computational overhead

## Project Structure

```
src/
├── smolvlm/                    # SmolVLM training
│   └── train_smolvlm_gate.py   # Main training script
├── data_prep/                  # Data generation and preparation
│   ├── gen_training_data.py
│   ├── gen_more_pixels_training_data_qwen.py
│   ├── get_training_dist.py
│   ├── prepare_textvqa.py
│   ├── create_low_res.py
│   └── convert_resolution_quality2parquet.py
└── utils/                      # Utility functions
    ├── res_stats2.py           # Resolution statistics
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

### Training SmolVLM Gate

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

# With feature layer averaging
python src/smolvlm/train_smolvlm_gate.py \
    --parquet data/hardness_data.parquet \
    --feat_layer middle \
    --feat_window 3 \
    --out ./checkpoints/smolvlm_averaged
```

### Data Preparation

```bash
# Generate training data
python src/data_prep/gen_training_data.py --input_file data/samples.jsonl

# Prepare existing VQA datasets
python src/data_prep/prepare_textvqa.py

# Analyze training data distribution
python src/utils/res_stats2.py --input data/training.parquet
```

## Models

### Supported SmolVLM Variants
- **SmolVLM-256M-Instruct**: Lightweight model, good for on-device deployment
- **SmolVLM-500M-Instruct**: Larger variant with improved accuracy

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
