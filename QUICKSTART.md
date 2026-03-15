# CARES Quick Start

## What is CARES?

CARES (Context-Aware Resolution Selection) learns whether images need high-resolution processing for accurate VLM predictions using SmolVLM, a lightweight direct prediction approach that's efficient and easy to deploy.

## 30-Second Overview

```bash
# Install
pip install -r requirements.txt

# Train SmolVLM model (efficient, on-device)
python src/smolvlm/train_smolvlm_gate.py \
    --parquet data/training.parquet \
    --model_name HuggingFaceTB/SmolVLM-256M-Instruct \
    --out ./checkpoint_smolvlm \
    --epochs 10 \
    --bsz 32
```

## SmolVLM Advantages

- **Fast**: Lightweight 256M-500M models
- **Efficient**: Freezes model weights, learns only classification head
- **Flexible**: Supports multiclass and binary classification
- **On-device**: Perfect for edge deployment

## File Organization

```
src/
├── smolvlm/         → train_smolvlm_gate.py (main training script)
├── data_prep/       → Dataset preparation scripts
└── utils/           → Analysis & utilities
```

## Key Concepts

### Training Data Format
Parquet file with columns:
- `question`: Text question
- `image_path` or `mid_path`: Path to image
- `hard`: Label (0/1 for binary, 0/1/2 for multiclass)

### Resolution Classes
- **0 (Low)**: Question answerable at low resolution (~384x384)
- **1 (Medium)**: Needs medium resolution (~512x512)
- **2 (High)**: Requires high resolution (~768x768+)

### Gate Output
Returns probability distribution over resolution classes for each image-question pair.

## Training SmolVLM in 5 Minutes

```bash
# 1. Prepare data
python src/data_prep/gen_training_data.py \
    --output data/training.parquet

# 2. Train SmolVLM with 256M model
python src/smolvlm/train_smolvlm_gate.py \
    --parquet data/training.parquet \
    --model_name HuggingFaceTB/SmolVLM-256M-Instruct \
    --out ./smolvlm_checkpoint \
    --epochs 10 \
    --bsz 32

# 3. Analyze results
python src/utils/res_stats2.py --input data/training.parquet
```

## Common Commands

```bash
# Resume training from checkpoint
python src/smolvlm/train_smolvlm_gate.py \
    --parquet data/training.parquet \
    --resume \
    --out ./smolvlm_checkpoint

# Use different learning rate
python src/smolvlm/train_smolvlm_gate.py \
    --parquet data/training.parquet \
    --lr 5e-4

# Analyze training distribution
python src/utils/res_stats2.py --input data/training.parquet

# Get confusion matrix
python src/utils/confusion.py --predictions results.json

# Binary classification (instead of 3-class)
python src/smolvlm/train_smolvlm_gate.py \
    --parquet data/training.parquet \
    --binary

# Use feature layer averaging
python src/smolvlm/train_smolvlm_gate.py \
    --parquet data/training.parquet \
    --feat_layer middle \
    --feat_window 3
```

## Expected Results

Typical performance on benchmark datasets:
- **Accuracy**: 75-85% for SmolVLM models
- **Computational savings**: 30-50% reduction vs. always high-res
- **Inference speed**: 5x faster than larger gate models on CPU

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `--bsz` or use `--create_low_res` |
| Slow training | Ensure GPU is being used, check CUDA availability |
| Bad results | Check data distribution with `res_stats2.py`, increase epochs |
| Missing images | Verify paths are absolute or relative from script dir |

## Next Steps

1. **Read full docs**: See README.md and DEVELOPING.md
2. **Set up data**: Follow scripts in src/data_prep/
3. **Train model**: Use train_smolvlm_gate.py with your dataset
4. **Analyze results**: Use res_stats2.py and confusion.py for evaluation

## API Usage

```python
from transformers import AutoProcessor, AutoTokenizer
from safetensors.torch import load_file
import torch

# Load trained SmolVLM
model = AutoModel.from_pretrained('./smolvlm_checkpoint')
processor = AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM-256M-Instruct')

# Prepare input
image = Image.open('image.jpg')
question = "What is in this image?"
inputs = processor(images=image, text=question, return_tensors='pt')

# Predict
with torch.no_grad():
    outputs = model(**inputs)

# Resolution prediction (0=low, 1=med, 2=high)
resolution_class = outputs.logits.argmax(dim=-1).item()
```

## Citation

[To be added when paper is published]

## Support

- 📖 Full documentation: See README.md and DEVELOPING.md
- 🐛 Issues: https://github.com/mkimhi/CARES/issues
- 💬 Discussions: https://github.com/mkimhi/CARES/discussions
