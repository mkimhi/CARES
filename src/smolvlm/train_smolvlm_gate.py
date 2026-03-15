"""
Train a small MLP gate on top of SmolVLM *intermediate* hidden states.

- Freezes all SmolVLM weights; learns only a classifier head.
- Uses text hidden states (which already cross-attend to the image).
- Pools over *non-image* tokens to form one feature per sample.
- Multiclass (default 3 classes: 0/1/2) or binary via --binary.
- NEW: --feat_layer INT|"middle" and --feat_window INT to average layers.

Parquet must contain columns: ["question", "mid_path", "hard"].

Tested with: transformers>=4.44, accelerate, datasets, sklearn, safetensors, torch, PIL, pandas
Models:
- HuggingFaceTB/SmolVLM-256M-Instruct  (~200–300M)
- HuggingFaceTB/SmolVLM-500M-Instruct  (~500M)
"""

import argparse, os, re, json
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoProcessor,
    AutoConfig,
    AutoModelForVision2Seq,AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import get_last_checkpoint
from safetensors.torch import load_file

# --- PIL safety for large / truncated images
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ------------------------ IO utils ------------------------
def load_image(path: str, target_size=None) -> Image.Image:
    """Robust loader: accepts a directory (picks first image) and fixes '_<num>.ext' suffixes."""
    if os.path.isdir(path):
        imgs = [f for f in os.listdir(path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))]
        if not imgs:
            raise FileNotFoundError(f"No image files in {path}")
        path = os.path.join(path, imgs[0])

    try:
        im = Image.open(path).convert("RGB")
    except FileNotFoundError:
        alt_path = re.sub(r'_(\d+)(?=\.(jpg|jpeg|png|bmp|webp)$)', '', path, flags=re.IGNORECASE)
        if os.path.exists(alt_path):
            im = Image.open(alt_path).convert("RGB")
        else:
            raise

    if target_size is not None:
        w, h = im.size
        tw, th = target_size
        new = Image.new("RGB", (tw, th), (0, 0, 0))
        new.paste(im, ((tw - w) // 2, (th - h) // 2))
        im = new
    return im


# ------------------------ Dataset ------------------------
# class HardnessSmolDS(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         parquet_path: str,
#         model_name: str,
#         split: str = "train",
#         val_frac: float = 0.1,
#         seed: int = 42,
#         binary: bool = False,
#         target_size: Tuple[int, int] = (784, 784),
#         use_chat_template: bool = True,
#     ):
#         df = pd.read_parquet(parquet_path)
#         train_df, val_df = train_test_split(df, test_size=val_frac, stratify=df["hard"], random_state=seed)
#         self.df = train_df if split == "train" else val_df

#         self.processor = AutoProcessor.from_pretrained(model_name)
#         self.tokenizer = self.processor.tokenizer
#         self.binary = binary
#         self.target_size = target_size
#         self.use_chat_template = use_chat_template

#         # Image token id resolution for SmolVLM
#         cfg = AutoConfig.from_pretrained(model_name)
#         self.image_token_id = getattr(cfg, "image_token_id", None)
#         if self.image_token_id is None:
#             # Fallbacks if config lacks it
#             image_tok = getattr(self.processor, "image_token", "<image>")
#             self.image_token_id = self.tokenizer.convert_tokens_to_ids(image_tok)
#             if self.image_token_id is None:
#                 # Last resort: add token
#                 self.tokenizer.add_tokens(["<image>"])
#                 self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
class HardnessSmolDS(torch.utils.data.Dataset):
    def __init__(
        self,
        parquet_path: str,
        model_name: str,
        split: str = "train",
        val_frac: float = 0.1,
        seed: int = 42,
        binary: bool = False,
        target_size: Tuple[int, int] | None = None,
        use_chat_template: bool = True,
    ):
        df = pd.read_parquet(parquet_path)
        train_df, val_df = train_test_split(df, test_size=val_frac, stratify=df["hard"], random_state=seed)
        self.df = train_df if split == "train" else val_df

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer
        self.binary = binary
        self.use_chat_template = use_chat_template

        # Resolve model config (works for SmolVLM and Granite-Docling, both idefics3)
        cfg = AutoConfig.from_pretrained(model_name)
        self.model_type = getattr(cfg, "model_type", None)


        # ---- NEW: infer default image size from vision_config if target_size not given ----
        if target_size is None:
            if self.model_type in {"qwen2_5_vl", "qwen2vl"}:
                # Qwen2.5-VL uses dynamic resolution via AutoProcessor; keep original image size
                target_size = None
            elif hasattr(cfg, "vision_config"):
                vc = cfg.vision_config
                edge = getattr(vc, "image_size", None)
                if edge is None and hasattr(vc, "max_image_size") and isinstance(vc.max_image_size, dict):
                    edge = vc.max_image_size.get("longest_edge", None)
                if edge is None and hasattr(vc, "size") and isinstance(vc.size, dict):
                    edge = vc.size.get("longest_edge", None)
                if edge is not None:
                    target_size = (edge, edge)
        if target_size is None:
            target_size = (784, 784)

        self.target_size = target_size

        # Image token id resolution for SmolVLM / Granite-Docling
        self.image_token_id = getattr(cfg, "image_token_id", None)
        if self.image_token_id is None:
            # Fallbacks if config lacks it (keeps your original logic)
            image_tok = getattr(self.processor, "image_token", "<image>")
            self.image_token_id = self.tokenizer.convert_tokens_to_ids(image_tok)
            if self.image_token_id is None:
                self.tokenizer.add_tokens(["<image>"])
                self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        question = str(row.question)
        img = load_image(row.mid_path, target_size=self.target_size)

        if self.use_chat_template and hasattr(self.processor, "apply_chat_template"):
            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
            ]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=False)
            enc = self.processor(text=[prompt], images=[img], return_tensors="pt")
        else:
            text = "<image>\n" + question
            enc = self.processor(text=[text], images=[img], return_tensors="pt")

        # squeeze batch dims produced by processor
        item = {k: (v.squeeze(0) if isinstance(v, torch.Tensor) else v[0]) for k, v in enc.items()}

        label_val = int(row.hard)
        if self.binary:
            label_val = 1 if label_val > 0 else 0
        item["labels"] = torch.tensor(label_val, dtype=torch.long)
        item["image_token_id"] = torch.tensor(self.image_token_id, dtype=torch.long)
        return item


# ------------------------ Collator ------------------------
# @dataclass
# class SmolCollator:
#     tokenizer: Any

#     def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
#         input_ids = [f["input_ids"] for f in features]
#         attention_mask = [f["attention_mask"] for f in features]
#         labels = torch.stack([f["labels"] for f in features])
#         pixel_values = torch.cat([f["pixel_values"].unsqueeze(0) for f in features], dim=0)
#         image_token_id = features[0].get("image_token_id", None)

#         max_len = max(x.size(0) for x in input_ids)
#         pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

#         def pad_seq(x, pad_val):
#             if x.size(0) == max_len:
#                 return x
#             out = torch.full((max_len,), pad_val, dtype=x.dtype)
#             out[: x.size(0)] = x
#             return out

#         input_ids = torch.stack([pad_seq(x, pad_id) for x in input_ids])
#         attention_mask = torch.stack([pad_seq(x, 0) for x in attention_mask])

#         batch = {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "pixel_values": pixel_values,
#             "labels": labels,
#         }
#         if image_token_id is not None:
#             batch["image_token_id"] = image_token_id
#         return batch
@dataclass
class SmolCollator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = torch.stack([f["labels"] for f in features])
        pixel_values = torch.cat([f["pixel_values"].unsqueeze(0) for f in features], dim=0)
        image_token_id = features[0].get("image_token_id", None)

        max_len = max(x.size(0) for x in input_ids)
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        def pad_seq(x, pad_val):
            if x.size(0) == max_len:
                return x
            out = torch.full((max_len,), pad_val, dtype=x.dtype)
            out[: x.size(0)] = x
            return out

        input_ids = torch.stack([pad_seq(x, pad_id) for x in input_ids])
        attention_mask = torch.stack([pad_seq(x, 0) for x in attention_mask])

        batch: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }
        if image_token_id is not None:
            batch["image_token_id"] = image_token_id

        # NEW: carry over any extra tensor fields (Qwen2.5-VL: image_grid_thw, pixel_values_videos, etc.)
        known = {"input_ids", "attention_mask", "pixel_values", "labels", "image_token_id"}
        for key in features[0].keys():
            if key in known:
                continue
            val0 = features[0][key]
            if isinstance(val0, torch.Tensor):
                batch[key] = torch.stack([f[key] for f in features])
            else:
                # non-tensor metadata (if any) is just passed as a list
                batch[key] = [f[key] for f in features]

        return batch

# ------------------------ Model (frozen SmolVLM + intermediate head) ------------------------
class SmolGate(nn.Module):
    def __init__(
        self,
        model_name: str,
        hidden: int = 256,
        num_classes: int = 3,
        feat_layer: Union[int, str] = "middle",
        feat_window: int = 0,
        dtype: torch.dtype = torch.float,
    ):
        super().__init__()
        self.backbone = AutoModelForImageTextToText.from_pretrained(model_name, dtype=dtype)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # Hidden/state sizes come from the text transformer config
        txt_cfg = self.backbone.config.text_config if hasattr(self.backbone.config, "text_config") else self.backbone.config
        self.num_layers = getattr(txt_cfg, "num_hidden_layers", None)
        if self.num_layers is None:
            raise ValueError("Could not resolve number of text layers from model config.")

        if isinstance(feat_layer, str):
            if feat_layer.lower() == "middle":
                self.feat_layer_index = self.num_layers // 2
            else:
                try:
                    self.feat_layer_index = int(feat_layer)
                except ValueError:
                    raise ValueError(f"Unrecognized feat_layer={feat_layer}")
        else:
            self.feat_layer_index = int(feat_layer)
        self.feat_layer_index = max(0, min(self.num_layers, self.feat_layer_index))
        self.feat_window = max(0, int(feat_window))

        d = getattr(txt_cfg, "hidden_size")
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes if num_classes > 1 else 1),
        )

    @torch.no_grad()
    def _select_hidden(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """Pick layer k or mean over a window [k-w, k+w]. hidden_states length = L+1 (0=embeddings, 1..L)."""
        L = len(hidden_states) - 1
        k = max(0, min(L, self.feat_layer_index))
        if self.feat_window == 0:
            return hidden_states[k]  # [B,T,D]
        lo = max(0, k - self.feat_window)
        hi = min(L, k + self.feat_window)
        stack = torch.stack([hidden_states[i] for i in range(lo, hi + 1)], dim=0)  # [W,B,T,D]
        return stack.mean(dim=0)

    @torch.no_grad()
    def extract_feat(self, input_ids, attention_mask, pixel_values, image_token_id=None, **vision_kwargs):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
            **vision_kwargs,  # NEW: e.g. image_grid_thw, pixel_values_videos, etc.
        )
        hs = self._select_hidden(out.hidden_states)  # [B,T,D]

        # pool over text tokens only (non-image) that are attended
        if image_token_id is not None:
            text_mask = (input_ids != image_token_id) & (attention_mask == 1)
        else:
            text_mask = (attention_mask == 1)
        fallback = (attention_mask == 1)
        use_mask = torch.where(text_mask.any(dim=1, keepdim=True), text_mask, fallback)
        lengths = use_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (hs * use_mask.unsqueeze(-1)).sum(dim=1) / lengths
        return pooled  # [B,D]

    def forward(
            self,
            input_ids,
            attention_mask,
            pixel_values,
            labels=None,
            image_token_id=None,
            **vision_kwargs,  # NEW: accept extra fields from the collator
        ):
        with torch.no_grad():
            feats = self.extract_feat(
                input_ids,
                attention_mask,
                pixel_values,
                image_token_id=image_token_id,
                **vision_kwargs,
            )
        logits = self.head(feats).squeeze(-1)

        loss = None
        if labels is not None:
            if logits.ndim == 1:  # binary
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, labels.float()
                )
            else:
                loss = nn.functional.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}


# ------------------------ Metrics ------------------------
def make_metrics(binary: bool):
    def _metrics(eval_pred):
        # HF may pass EvalPrediction or (preds, labels)
        if isinstance(eval_pred, tuple) or isinstance(eval_pred, list):
            preds, labels = eval_pred
        else:
            preds, labels = eval_pred.predictions, eval_pred.label_ids
        logits = preds
        if binary:
            probs = torch.sigmoid(torch.tensor(logits)).numpy()
            predicted = (probs > 0.5).astype(int)
            labels_np = np.array(labels)
            return {
                "auc": float(roc_auc_score(labels_np, probs)),
                "accuracy": float(accuracy_score(labels_np, predicted)),
                "precision": float(precision_score(labels_np, predicted, zero_division=0)),
                "recall": float(recall_score(labels_np, predicted, zero_division=0)),
            }
        else:
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
            predicted = probs.argmax(axis=-1)
            labels_np = np.array(labels)
            try:
                auc = float(roc_auc_score(labels_np, probs, multi_class="ovr"))
            except ValueError:
                auc = float("nan")
            return {
                "auc": auc,
                "accuracy": float(accuracy_score(labels_np, predicted)),
                "precision_macro": float(precision_score(labels_np, predicted, average="macro", zero_division=0)),
                "recall_macro": float(recall_score(labels_np, predicted, average="macro", zero_division=0)),
            }
    return _metrics


# ------------------------ CLI / Train ------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--model_name", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    ap.add_argument("--out", default="./gate_vlm_smol_intermediate")
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gate_hidden", type=int, default=256)
    ap.add_argument("--binary", action="store_true")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume_from", type=str, default=None)
    # NEW:
    ap.add_argument("--target_size", type=int, nargs=2, metavar=("W", "H"),
                    default=None, help="Optional image size override (width height)")
    ap.add_argument("--feat_layer", default="middle", help='Layer index: int (0..num_layers) or "middle"')
    ap.add_argument("--feat_window", type=int, default=0, help="Average layers [k-w, k+w]. Default 0.")
    ap.add_argument("--no_chat_template", action="store_true", help="Disable chat template; prepend <image> manually")
    return ap.parse_args()


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_ds = HardnessSmolDS(
        args.parquet, args.model_name, "train", args.val_frac, args.seed,
        binary=args.binary,
        target_size=tuple(args.target_size) if args.target_size is not None else None,
        use_chat_template=not args.no_chat_template,
    )
    val_ds = HardnessSmolDS(
        args.parquet, args.model_name, "val", args.val_frac, args.seed,
        binary=args.binary,
        target_size=tuple(args.target_size) if args.target_size is not None else None,
        use_chat_template=not args.no_chat_template,
    )
    collate = SmolCollator(train_ds.tokenizer)

    num_classes = 1 if args.binary else 3
    model = SmolGate(
        args.model_name,
        hidden=args.gate_hidden,
        num_classes=num_classes,
        feat_layer=args.feat_layer,
        feat_window=args.feat_window,
        dtype=torch.float,  # bfloat16 ok for inference, but we keep float to be safe for head training
    )

    metric_name = "auc" if args.binary else "accuracy"
    training_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        bf16=True,  # compute in bf16 where possible
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        greater_is_better=True,
        report_to="none",
        save_safetensors=False,
        label_smoothing_factor=0.0,
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
        metrics = trainer.evaluate()
        os.makedirs(args.out, exist_ok=True)
        out_path = os.path.join(args.out, "eval_metrics.json")
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print("Eval-only metrics:", metrics)
        print(f"Saved eval metrics to {out_path}")
        return

    resume_ckpt = None
    if args.resume_from:
        resume_ckpt = args.resume_from
    elif args.resume and os.path.isdir(args.out) and get_last_checkpoint(args.out) is not None:
        resume_ckpt = get_last_checkpoint(args.out)

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(args.out)
    
    metrics = trainer.evaluate()
    metrics["best_model_checkpoint"] = trainer.state.best_model_checkpoint

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "eval_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Final eval metrics:", metrics)
    print(f"Saved final eval metrics to {out_path}")


if __name__ == "__main__":
    main()


"""
SMOL:

middle - 63.3%
last - 62.3%

G-docling - middle:
Final eval metrics: {'eval_loss': 0.8530066013336182, 'eval_auc': 0.7667570296006302, 'eval_accuracy': 0.6079580129378738, 'eval_precision_macro': 0.6078310212541055, 'eval_recall_macro': 0.5665985707314106, 'eval_runtime': 360.5151, 'eval_samples_per_second': 22.726, 'eval_steps_per_second': 0.18, 'epoch': 6.0, 'best_model_checkpoint': './gate_docling/checkpoint-3000'}
"""