# CARES Quick Start

## What is CARES?

CARES (Context-Aware Resolution Selection) learns whether images need high-resolution processing for accurate VLM predictions. It provides two approaches:
- **Gates**: Separate classifiers predicting resolution needs
- **SmolVLM**: Direct prediction using lightweight model

## 30-Second Overview

```bash
# Install
pip install -r requirements.txt

# Train a gate (predicts if high-res needed)
python src/gates/train_gate_siglip.py --out ./checkpoint_gate

# Train SmolVLM gate (faster, on-device)
python src/smolvlm/train_smolvlm_gate.py \
    --parquet data/training.parquet \
    --out ./checkpoint_smolvlm
```

## Which approach should I use?

| Aspect | Gate | SmolVLM |
|--------|------|---------|
| **Speed** | Slower | ⚡ Fast |
| **Accuracy** | Higher | Good |
| **Model size** | 400M+ | 256M-500M |
| **Flexibility** | Many base models | Fixed to SmolVLM |
| **Use case** | Production accuracy | Mobile/edge devices |

## File Organization

```
src/
├── gates/           → train_gate_*.py (SigLIP, VLM variants)
├── smolvlm/         → train_smolvlm_gate.py
├── data_prep/       → Dataset preparation scripts
├── inference/       → Inference & evaluation
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

## Training a Gate in 5 Minutes

```bash
# 1. Prepare data
python src/data_prep/gen_training_data.py \
    --output data/training.parquet

# 2. Train gate with SigLIP features
python src/gates/train_gate_siglip.py \
    --parquet data/training.parquet \
    --out ./gate_checkpoint \
    --epochs 10 \
    --bsz 32

# 3. Evaluate
python src/inference/run_cares_on_gv_data.py \
    --checkpoint ./gate_checkpoint \
    --test_file data/test.json
```

## Training SmolVLM Gate

```bash
python src/smolvlm/train_smolvlm_gate.py \
    --parquet data/training.parquet \
    --model_name HuggingFaceTB/SmolVLM-256M-Instruct \
    --out ./smolvlm_checkpoint \
    --epochs 20 \
    --bsz 64 \
    --lr 1e-4
```

## Common Commands

```bash
# Resume training from checkpoint
python src/gates/train_gate_siglip.py --out ./checkpoint --resume

# Use different learning rate
python src/gates/train_gate_siglip.py --lr 5e-4

# Analyze training distribution
python src/utils/res_stats2.py --input data/training.parquet

# Get confusion matrix
python src/utils/confusion.py --predictions results.json

# Binary classification (instead of 3-class)
python src/smolvlm/train_smolvlm_gate.py --binary
```

## Expected Results

Typical performance on benchmark datasets:
- **Accuracy**: 75-85% for gate models
- **Computational savings**: 30-50% reduction vs. always high-res
- **SmolVLM gate**: Slightly lower accuracy but 5x faster inference

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `--bsz` or use `--create_low_res` |
| Slow training | Use SmolVLM approach instead of gate |
| Bad results | Check data distribution with `res_stats2.py` |
| Missing images | Verify paths are absolute or relative from script dir |

## Next Steps

1. **Read full docs**: See README.md and DEVELOPING.md
2. **Set up data**: Follow scripts in src/data_prep/
3. **Choose approach**: Gate for accuracy, SmolVLM for speed
4. **Train model**: Run appropriate training script
5. **Evaluate**: Use inference scripts on your datasets

## API Usage

```python
from transformers import AutoProcessor, AutoModel
import torch

# Load trained gate
model = AutoModel.from_pretrained('./checkpoint')
processor = AutoProcessor.from_pretrained('google/siglip-so400m-patch14-384')

# Predict
image = Image.open('image.jpg')
text = "What is in this image?"
inputs = processor(images=image, text=text, return_tensors='pt')
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
