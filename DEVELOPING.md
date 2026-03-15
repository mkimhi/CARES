# Development Guide

## Quick Start for Development

### Setting up development environment

```bash
# Clone and install
git clone https://github.com/mkimhi/CARES.git
cd CARES
pip install -r requirements.txt

# For development with additional tools
pip install pre-commit black flake8 pytest
```

## Module Overview

### SmolVLM (`src/smolvlm/`)
Direct prediction using a lightweight model:

- **train_smolvlm_gate.py**: MLP classifier on frozen SmolVLM intermediate representations
  - Freezes all model weights
  - Learns only the classification head
  - Supports multiclass (default 3: low/medium/high resolution needed)
  - Can average multiple layer outputs for robustness

**Key features:**
- Configurable feature layer (`--feat_layer`)
- Layer window averaging (`--feat_window`)
- Binary and multiclass modes

### Data Preparation (`src/data_prep/`)
Scripts for generating and preparing training data:

- **gen_training_data.py**: Main script for creating hardness labels
- **gen_more_pixels_training_data_qwen.py**: Qwen-specific data generation
- **prepare_textvqa.py**: Prepare TextVQA dataset
- **get_training_dist.py**: Analyze distribution of training examples
- **create_low_res.py**: Generate low-resolution versions of images
- **convert_resolution_quality2parquet.py**: Convert data to parquet format

### Utilities (`src/utils/`)
Helper functions and analysis tools:

- **res_stats2.py**: Resolution statistics and analysis
- **confusion.py**: Confusion matrix visualization
- **sft_cares.py**: Supervised fine-tuning utilities
- **upload_lora.py**: Upload LoRA adapters
- **download.py**: Download datasets
- **valid_res.py**: Validate resolution predictions

## Common Workflows

### Training SmolVLM Gate

```bash
# 256M model
python src/smolvlm/train_smolvlm_gate.py \
    --parquet data/hardness_data.parquet \
    --model_name HuggingFaceTB/SmolVLM-256M-Instruct \
    --out ./checkpoints/smolvlm_256m

# 500M model with binary classification
python src/smolvlm/train_smolvlm_gate.py \
    --parquet data/hardness_data.parquet \
    --model_name HuggingFaceTB/SmolVLM-500M-Instruct \
    --binary \
    --out ./checkpoints/smolvlm_500m_binary

# With custom feature layer selection
python src/smolvlm/train_smolvlm_gate.py \
    --parquet data/hardness_data.parquet \
    --feat_layer middle \
    --feat_window 3
```

### Preparing Data

```bash
# Generate training data
python src/data_prep/gen_training_data.py \
    --dataset textvqa \
    --output data/training_data.parquet

# Prepare existing dataset
python src/data_prep/prepare_textvqa.py

# Create low-resolution versions
python src/data_prep/create_low_res.py \
    --input data/high_res_images \
    --output data/low_res_images
```

## Code Style

- Use type hints for function signatures
- Add docstrings to public functions
- Keep functions focused and single-purpose
- Use descriptive variable names

## Debugging Tips

### Memory issues with large images
The code includes PIL safety settings:
```python
Image.MAX_IMAGE_PIXELS = None  # Allow very large images
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Handle corrupted images
```

If you hit OOM errors:
1. Reduce batch size (`--bsz`)
2. Reduce image size or use `--create_low_res`
3. Check for leaks in data loading pipeline

### Checkpoint issues
- Always use `--resume` flag when retraining
- Checkpoints include model weights, optimizer state, and training metrics
- Use `get_last_checkpoint()` to find latest checkpoint

### Dataset issues
- Verify parquet format with `pd.read_parquet()`
- Check that images paths are absolute or relative from correct dir
- Validate label distribution with `get_training_dist.py`

## Contributing

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes and test thoroughly
3. Commit with clear messages: `git commit -m "Add feature X"`
4. Push and create a pull request

## Testing

```bash
# Run inference on a single sample
python src/inference/run_cares_on_gv_data.py \
    --checkpoint checkpoints/gate_siglip \
    --test-sample data/sample.json
```

## Performance Profiling

Monitor training with Weights & Biases:
```bash
# Enable during training
python src/gates/train_gate_siglip.py \
    --out checkpoints/gate \
    --use_wandb  # if supported by script
```

Check resolution statistics:
```bash
python src/utils/res_stats2.py --predictions predictions.json
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce batch size, use gradient checkpointing |
| Slow data loading | Use cached datasets, increase num_workers |
| Bad predictions | Check data distribution, verify labels, increase epochs |
| Missing checkpoints | Ensure output directory has write permissions |

## References

- SmolVLM: [HuggingFaceTB/SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)
- SigLIP: [google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384)
- Datasets: TextVQA, DocVQA, ChartQA, InfographicVQA
