"""
Train a small MLP gate on top of Qwen2.5-VL hidden states.
- Freezes Qwen weights and learns only a classifier head.
- Uses the text hidden states (which already cross-attend to the image) and
  pools them over *non-image* tokens to form a single feature per sample.
- Supports multiclass (default 3 classes: 0/1/2) or binary via --binary.

Example:
python qwen_gate_trainer.py \
  --parquet hardness_data_mix.parquet \
  --model_name Qwen/Qwen2.5-VL-3B-Instruct \
  --out ./gate_vlm_qwen \
  --lr 1e-3 --bsz 128 --epochs 6 --gate_hidden 256

Requires: transformers>=4.44, accelerate, datasets, sklearn, safetensors, torch, PIL, pandas
"""
import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
import os, re
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import get_last_checkpoint

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms.functional as TF
#QWEN = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.float)



def load_image(path: str, target_size=None) -> Image.Image:
    # If it's a directory, pick the first image
    if os.path.isdir(path):
        imgs = [f for f in os.listdir(path) if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".webp"))]
        if not imgs:
            raise FileNotFoundError(f"No image files in {path}")
        path = os.path.join(path, imgs[0])

    # Try opening the original path
    try:
        im = Image.open(path).convert("RGB")
    except FileNotFoundError:
        # Try removing "_<num>" before the extension
        alt_path = re.sub(r'_(\d+)(?=\.(jpg|jpeg|png|bmp|webp)$)', '', path, flags=re.IGNORECASE)
        if os.path.exists(alt_path):
            im = Image.open(alt_path).convert("RGB")
        else:
            raise

    # Optional resize with padding
    if target_size is not None:
        w, h = im.size
        tw, th = target_size
        new = Image.new("RGB", (tw, th), (0, 0, 0))
        new.paste(im, ((tw - w)//2, (th - h)//2))
        im = new

    return im


# then in dataset __getitem__, collect current max size if needed and pad images in collator
# simplest: pad in collator to max in batch



# def load_image(path: str) -> Image.Image:
#     if os.path.isdir(path):
#         imgs = [f for f in os.listdir(path) if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".webp"))]
#         if not imgs:
#             raise FileNotFoundError(f"No image files in {path}")
#         path = os.path.join(path, imgs[0])
#     im = Image.open(path).convert("RGB")
#     return im

# ------------------------ Dataset ------------------------
class HardnessQwenDS(torch.utils.data.Dataset):
    def __init__(self, parquet_path: str, model_name: str, split: str = "train", val_frac: float = 0.1, seed: int = 42, binary: bool = False):
        df = pd.read_parquet(parquet_path)
        train_df, val_df = train_test_split(df, test_size=val_frac, stratify=df["hard"], random_state=seed)
        self.df = train_df if split == "train" else val_df
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.processor.num_additional_image_tokens = 1
        self.processor.num_additional_tokens=1
        self.tokenizer = self.processor.tokenizer
        self.binary = binary
        # cache image token id
        img_tok = getattr(self.processor, "image_token", "<image>")
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(img_tok)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        # Compose text with explicit <image> so the processor builds visual tokens
        #text = f"<image>\n{row.question}"
        img_tok = getattr(self.processor, "image_token", "<image>")   # << use processor's token
        text = f"{img_tok}\n{row.question}"                           # << instead of hardcoded "<image>"

        img = load_image(row.mid_path, target_size=(784,784))

        enc = self.processor(text=[text], images=[img], return_tensors="pt")
        # squeeze batch dim from processor
        item = {k: (v.squeeze(0) if isinstance(v, torch.Tensor) else v[0]) for k, v in enc.items()}
        # labels
        item = {k: (v.squeeze(0) if isinstance(v, torch.Tensor) else v[0]) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(row.hard) if not self.binary else (1 if int(row.hard) > 0 else 0), dtype=torch.long)
        item["image_token_id"] = torch.tensor(self.image_token_id, dtype=torch.long)
        return item

# ------------------------ Collator ------------------------
@dataclass
class QwenCollator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Pad input_ids/attention_mask; stack pixel_values + image_grid_thw; passthrough modalities
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = torch.stack([f["labels"] for f in features])
        pixel_values = torch.cat([f["pixel_values"].unsqueeze(0) for f in features], dim=0)
        image_grid_thw = torch.cat([f["image_grid_thw"].unsqueeze(0) for f in features], dim=0)
        #modalities = [f["modalities"] for f in features]
        image_token_id = features[0]["image_token_id"]  # same for all

        # max_w = max(f["pixel_values"].size(-1) for f in features)
        # max_h = max(f["pixel_values"].size(-2) for f in features)
        # padded_pixels = []
        # for f in features:
        #     pv = f["pixel_values"]
        #     c, h, w = pv.shape
        #     new = torch.zeros((c, max_h, max_w), dtype=pv.dtype)
        #     y0 = (max_h - h)//2
        #     x0 = (max_w - w)//2
        #     new[:, y0:y0+h, x0:x0+w] = pv
        #     padded_pixels.append(new)
        # pixel_values = torch.stack(padded_pixels)

        # pad text
        max_len = max(x.size(0) for x in input_ids)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        def pad_seq(x, pad_val):
            if x.size(0) == max_len:
                return x
            out = torch.full((max_len,), pad_val, dtype=x.dtype)
            out[: x.size(0)] = x
            return out
        input_ids = torch.stack([pad_seq(x, pad_id) for x in input_ids])
        attention_mask = torch.stack([pad_seq(x, 0) for x in attention_mask])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            #"modalities": ['image'],
            "labels": labels,
            "image_token_id": image_token_id,
        }

# ------------------------ Model (frozen Qwen + head) ------------------------
class QwenGate(nn.Module):
    def __init__(self, model_name: str, hidden: int = 256, num_classes: int = 3):
        super().__init__()
        self.qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float
        )
        self.qwen.eval()
        for p in self.qwen.parameters():
            p.requires_grad_(False)
        d = self.qwen.config.hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, num_classes if num_classes > 1 else 1)
        )

    @torch.no_grad()
    def extract_feat(self, input_ids, attention_mask, pixel_values, image_grid_thw, image_token_id):
        out = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        # out = QWEN(input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     pixel_values=pixel_values,
        #     image_grid_thw=image_grid_thw,
        #     output_hidden_states=True,
        #     use_cache=False,
        #     return_dict=True,
        # )
        # last hidden states [B, T, D]
        hs = out.hidden_states[-1]
        # mask out image tokens; keep only textual tokens that are attended
        text_mask = (input_ids != image_token_id) & (attention_mask == 1)
        # avoid division by zero: if a sequence is all image tokens (shouldn't happen), fallback to mean over all attended tokens
        lengths = text_mask.sum(dim=1, keepdim=True)
        fallback = (attention_mask == 1)
        lengths = torch.where(lengths > 0, lengths, fallback.sum(dim=1, keepdim=True))
        use_mask = torch.where(text_mask.any(dim=1, keepdim=True), text_mask, fallback)
        pooled = (hs * use_mask.unsqueeze(-1)).sum(dim=1) / lengths.clamp(min=1)
        return pooled  # [B, D]

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw, labels=None, image_token_id=None):
        with torch.no_grad():
            feats = self.extract_feat(input_ids, attention_mask, pixel_values, image_grid_thw, image_token_id)
        logits = self.head(feats).squeeze(-1)
        loss = None
        if labels is not None:
            if logits.ndim == 1:  # binary
                loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
            elif logits.ndim == 2 and logits.size(-1) > 1:
                loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}

# ------------------------ Metrics ------------------------

def make_metrics(binary: bool):
    def _metrics(eval_pred):
        logits, labels = eval_pred
        if binary:
            probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
            preds = (probs > 0.5).astype(int)
            labels_np = np.array(labels)
            return {
                "auc": float(roc_auc_score(labels_np, probs)),
                "accuracy": float(accuracy_score(labels_np, preds)),
                "precision": float(precision_score(labels_np, preds, zero_division=0)),
                "recall": float(recall_score(labels_np, preds, zero_division=0)),
            }
        else:
            probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
            preds = probs.argmax(axis=-1)
            labels_np = np.array(labels)
            try:
                auc = float(roc_auc_score(labels_np, probs, multi_class="ovr"))
            except ValueError:
                auc = float("nan")
            return {
                "auc": auc,
                "accuracy": float(accuracy_score(labels_np, preds)),
                "precision_macro": float(precision_score(labels_np, preds, average="macro", zero_division=0)),
                "recall_macro": float(recall_score(labels_np, preds, average="macro", zero_division=0)),
            }
    return _metrics

# ------------------------ CLI / Train ------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--out", default="./gate_vlm_qwen")
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gate_hidden", type=int, default=256)
    ap.add_argument("--binary", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume_from", type=str, default=None)
    ap.add_argument("--eval_only", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True

    train_ds = HardnessQwenDS(args.parquet, args.model_name, "train", args.val_frac, args.seed, binary=args.binary)
    val_ds = HardnessQwenDS(args.parquet, args.model_name, "val", args.val_frac, args.seed, binary=args.binary)
    collate = QwenCollator(train_ds.tokenizer)

    num_classes = 1 if args.binary else 3
    model = QwenGate(args.model_name, hidden=args.gate_hidden, num_classes=num_classes)

    metric_name = "auc" if args.binary else "accuracy"

    training_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=20,
        eval_strategy="steps", #"steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_safetensors=False,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        greater_is_better=True,
        report_to='none',
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
        compute_metrics=make_metrics(args.binary),
    )

    if args.eval_only:
        print(trainer.evaluate())
        return

    resume_ckpt = None
    if args.resume_from:
        resume_ckpt = args.resume_from
    elif args.resume and os.path.isdir(args.out) and get_last_checkpoint(args.out) is not None:
        resume_ckpt = get_last_checkpoint(args.out)

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(args.out)

if __name__ == "__main__":
    main()
