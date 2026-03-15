# CARES Quick Start

## What is CARES?

CARES (Context-Aware Resolution Selection) learns whether images need high-resolution processing for accurate VLM predictions. It provides two equally valid approaches:

1. **SmolVLM Classifier**: Fast, efficient classifier on frozen SmolVLM features
2. **Granite-Docling Autoregressive**: Fine-tuned autoregressive model for direct prediction

## 30-Second Overview

```bash
# Install
pip install -r requirements.txt

# Choose your approach:

# Option 1: SmolVLM Classifier (lightweight, on-device)
python src/training/train_smolvlm_gate.py \
    --parquet data/training.parquet \
    --model_name HuggingFaceTB/SmolVLM-256M-Instruct \
    --out ./checkpoint_smolvlm \
    --epochs 10

# Option 2: Granite-Docling Autoregressive (production, interpretable)
python src/training/train_granite_sft.py \
    --parquet data/training.parquet \
    --output_dir ./checkpoint_granite \
    --num_epochs 3 \
    --use_lora
```

## Approach Comparison

| Aspect | SmolVLM Classifier | Granite-Docling SFT |
|--------|-------------------|-------------------|
| **Model Size** | 256M-500M | Foundation model |
| **Training** | Frozen features + MLP | Full SFT + LoRA |
| **Inference** | Classification head | Autoregressive |
| **Speed** | ⚡⚡⚡ Fast | ⚡⚡ Medium |
| **Hosting** | On-device, edge | Production servers |
| **Interpretability** | Class scores | Direct text output |
| **Best for** | Low-latency, edge | Production applications |

## File Organization

```
src/
├── training/        → Both training approaches
│   ├── train_smolvlm_gate.py (SmolVLM classifier)
│   └── train_granite_sft.py (Granite-Docling SFT)
└── utils/           → Analysis & utilities (res_stats2, confusion, etc.)
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

## Training SmolVLM Classifier

```bash
# Train SmolVLM with 256M model
# (Requires parquet file with columns: question, mid_path, hard)
python src/training/train_smolvlm_gate.py \
    --parquet data/training.parquet \
    --model_name HuggingFaceTB/SmolVLM-256M-Instruct \
    --out ./smolvlm_checkpoint \
    --epochs 10 \
    --bsz 32

# Analyze results
python src/utils/res_stats2.py --input data/training.parquet
```

## Training Granite-Docling SFT

```bash
# Train Granite with SFT + LoRA
# (Requires parquet file with columns: question, mid_path, hard)
python src/training/train_granite_sft.py \
    --parquet data/training.parquet \
    --output_dir ./granite_checkpoint \
    --num_epochs 3 \
    --batch_size 32 \
    --use_lora

# Analyze results
python src/utils/res_stats2.py --input data/training.parquet
```

## Common Commands

```bash
# SmolVLM: Resume training from checkpoint
python src/training/train_smolvlm_gate.py \
    --parquet data/training.parquet \
    --resume \
    --out ./smolvlm_checkpoint

# SmolVLM: Binary classification (instead of 3-class)
python src/training/train_smolvlm_gate.py \
    --parquet data/training.parquet \
    --binary

# Granite: Fine-tune with custom learning rate
python src/training/train_granite_sft.py \
    --parquet data/training.parquet \
    --learning_rate 5e-5

# Analyze training distribution
python src/utils/res_stats2.py --input data/training.parquet

# Get confusion matrix
python src/utils/confusion.py --predictions results.json

# Upload LoRA adapters (for Granite)
python src/utils/upload_lora.py --checkpoint ./granite_checkpoint
```

## Expected Results

### SmolVLM Classifier
- **Accuracy**: 75-85%
- **Computational savings**: 30-50% vs. always high-res
- **Inference speed**: 5x faster than larger models on CPU

### Granite-Docling SFT
- **Accuracy**: 78-88%
- **Computational savings**: 30-50% vs. always high-res
- **Easy deployment**: Self-contained model, no separate classifier

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `--bsz` or use `--create_low_res` |
| Slow training | Ensure GPU is being used, check CUDA availability |
| Bad results | Check data distribution with `res_stats2.py`, increase epochs |
| Missing images | Verify paths are absolute or relative from script dir |

## Next Steps

1. **Read full docs**: See README.md and DEVELOPING.md
2. **Prepare training data**: Create parquet with columns: question, mid_path, hard
3. **Choose approach**: SmolVLM for speed, Granite for production
4. **Train model**: Run train_smolvlm_gate.py or train_granite_sft.py
5. **Analyze results**: Use res_stats2.py and confusion.py for evaluation

## API Usage - SmolVLM

```python
from transformers import AutoProcessor, AutoModel
import torch

# Load trained SmolVLM classifier
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

## API Usage - Granite-Docling

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

# Load trained Granite model with LoRA adapters
model = AutoModelForVision2Seq.from_pretrained('./granite_checkpoint')
processor = AutoProcessor.from_pretrained('./granite_checkpoint')

# Prepare input
image = Image.open('image.jpg')
question = "What resolution is sufficient for this question?"
inputs = processor(images=image, text=question, return_tensors='pt')

# Predict
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=10)

# Decode output
resolution_prediction = processor.decode(outputs[0])
```

## Citation

[To be added when paper is published]

## Support

- 📖 Full documentation: See README.md and DEVELOPING.md
- 🐛 Issues: https://github.com/mkimhi/CARES/issues
- 💬 Discussions: https://github.com/mkimhi/CARES/discussions
