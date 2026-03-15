"""
Train a small MLP gate on top of Qwen2.5-VL *intermediate* hidden states.

- Freezes all Qwen weights; learns only a classifier head.
- Uses text hidden states (which already cross-attend to the image).
- Pools over *non-image* tokens to form one feature per sample.
- Multiclass (default 3 classes: 0/1/2) or binary via --binary.
- NEW: --feat_layer INT|"middle" and --feat_window INT to average layers.

Parquet must contain columns: ["question", "mid_path", "hard"].

Requires: transformers>=4.44, accelerate, datasets, sklearn, safetensors, torch, PIL, pandas
"""

import argparse, os, re
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint
from safetensors.torch import load_file


# --- PIL safety for large / truncated images
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_tokenizer(model_name='Qwen/Qwen2.5-VL-3B-Instruct'):
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer
    # Ensure image token is present; some processors let you tweak these:
    processor.num_additional_image_tokens = getattr(processor, "num_additional_image_tokens", 1)
    processor.num_additional_tokens = getattr(processor, "num_additional_tokens", 1)
    return tokenizer

def load_gate_model(model_name='Qwen/Qwen2.5-VL-3B-Instruct',
                    load_from='/proj/mmfm/data/gate_vlm_qwen_mid/pytorch_model.bin'):
    gate = QwenGate(model_name, feat_layer='middle')

    if load_from is not None:
        safetensor_path = os.path.join(load_from, "model.safetensors")
        bin_path = os.path.join(load_from, "pytorch_model.bin")

        if os.path.exists(safetensor_path):
            state_dict = load_file(safetensor_path)
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError("No model.safetensors or pytorch_model.bin found")
        gate.load_state_dict(state_dict)
    return gate



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
class HardnessQwenDS(torch.utils.data.Dataset):
    def __init__(
        self,
        parquet_path: str,
        model_name: str,
        split: str = "train",
        val_frac: float = 0.1,
        seed: int = 42,
        binary: bool = False,
        target_size=(784, 784),
    ):
        df = pd.read_parquet(parquet_path)
        train_df, val_df = train_test_split(df, test_size=val_frac, stratify=df["hard"], random_state=seed)
        self.df = train_df if split == "train" else val_df

        self.processor = AutoProcessor.from_pretrained(model_name)
        # Ensure image token is present; some processors let you tweak these:
        self.processor.num_additional_image_tokens = getattr(self.processor, "num_additional_image_tokens", 1)
        self.processor.num_additional_tokens = getattr(self.processor, "num_additional_tokens", 1)
        self.tokenizer = self.processor.tokenizer
        self.binary = binary
        self.target_size = target_size

        img_tok = getattr(self.processor, "image_token", "<image>")
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(img_tok)
        self.image_token_text = img_tok

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        text = f"{self.image_token_text}\n{row.question}"  # prompts VL to insert visual tokens
        img = load_image(row.mid_path, target_size=self.target_size)

        enc = self.processor(text=[text], images=[img], return_tensors="pt")
        item = {k: (v.squeeze(0) if isinstance(v, torch.Tensor) else v[0]) for k, v in enc.items()}

        label_val = int(row.hard)
        if self.binary:
            label_val = 1 if label_val > 0 else 0
        item["labels"] = torch.tensor(label_val, dtype=torch.long)

        #item["image_token_id"] = torch.tensor(self.image_token_id, dtype=torch.long)
        #did not work for cauldrone3.parquet
        item["image_token_id"] = int(self.image_token_id)

        return item


# ------------------------ Collator ------------------------
@dataclass
class QwenCollator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = torch.stack([f["labels"] for f in features])
        pixel_values = torch.cat([f["pixel_values"].unsqueeze(0) for f in features], dim=0)
        image_grid_thw = torch.cat([f["image_grid_thw"].unsqueeze(0) for f in features], dim=0)
        image_token_id = features[0]["image_token_id"]

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
            "labels": labels,
            "image_token_id": image_token_id,
        }


# ------------------------ Model (frozen Qwen + intermediate head) ------------------------
class QwenGate(nn.Module):
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
        self.qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype)
        self.qwen.eval()
        for p in self.qwen.parameters():
            p.requires_grad_(False)

        # hidden_states length = L+1 (0=embeddings, 1..L)
        self.num_layers = self.qwen.config.num_hidden_layers
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

        d = self.qwen.config.hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes if num_classes > 1 else 1),
        )

    @torch.no_grad()
    def _select_hidden(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """Pick layer k or mean over a window [k-w, k+w]."""
        L = len(hidden_states) - 1  # number of transformer blocks
        k = max(0, min(L, self.feat_layer_index))
        if self.feat_window == 0:
            return hidden_states[k]  # [B,T,D]
        lo = max(0, k - self.feat_window)
        hi = min(L, k + self.feat_window)
        stack = torch.stack([hidden_states[i] for i in range(lo, hi + 1)], dim=0)  # [W,B,T,D]
        return stack.mean(dim=0)

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
        hs = self._select_hidden(out.hidden_states)  # [B,T,D]

        # pool over text tokens only (non-image) that are attended
        text_mask = (input_ids != image_token_id) & (attention_mask == 1)  # [B,T]
        fallback = (attention_mask == 1)
        use_mask = torch.where(text_mask.any(dim=1, keepdim=True), text_mask, fallback)
        lengths = use_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (hs * use_mask.unsqueeze(-1)).sum(dim=1) / lengths
        return pooled  # [B,D]

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw, labels=None, image_token_id=None):
        with torch.no_grad():
            feats = self.extract_feat(input_ids, attention_mask, pixel_values, image_grid_thw, image_token_id)
        logits = self.head(feats).squeeze(-1)
        loss = None
        if labels is not None:
            if logits.ndim == 1:  # binary
                loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
            else:
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
    ap.add_argument("--out", default="./gate_vlm_qwen_intermediate")
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
    ap.add_argument("--feat_layer", default="middle", help='Layer index: int (0..num_layers) or "middle"')
    ap.add_argument("--feat_window", type=int, default=0, help="Average layers [k-w, k+w]. Default 0.")
    return ap.parse_args()


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_ds = HardnessQwenDS(args.parquet, args.model_name, "train", args.val_frac, args.seed, binary=args.binary)
    val_ds = HardnessQwenDS(args.parquet, args.model_name, "val", args.val_frac, args.seed, binary=args.binary)
    collate = QwenCollator(train_ds.tokenizer)

    num_classes = 1 if args.binary else 3
    model = QwenGate(
        args.model_name,
        hidden=args.gate_hidden,
        num_classes=num_classes,
        feat_layer=args.feat_layer,
        feat_window=args.feat_window,
        dtype=torch.float,  # set to bfloat16 if you only run inference on Qwen (we freeze anyway)
    )

    metric_name = "auc" if args.binary else "accuracy"
    training_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        #bf16=True,  # compute in bf16 where possible
        logging_steps=20,
        eval_strategy="epoch",
        save_total_limit=2,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        greater_is_better=True,
        report_to="none",
        save_safetensors=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
        compute_metrics=make_metrics(args.binary),
    )



    resume_ckpt = None
    if args.resume_from:
        resume_ckpt = args.resume_from
    elif args.resume and os.path.isdir(args.out) and get_last_checkpoint(args.out) is not None:
        resume_ckpt = get_last_checkpoint(args.out)


    if args.eval_only:
        print(trainer.evaluate())
        return

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(args.out)
    trainer.evaluate()


if __name__ == "__main__":
    main()
    


"""


13 - {'eval_loss': 1.1978527307510376, 'eval_auc': 0.41633144105220765, 'eval_accuracy': 0.21860124496521421, 'eval_precision_macro': 0.21315751971043853, 'eval_recall_macro': 0.2797625003370121, 'eval_runtime': 716.9619, 'eval_samples_per_second': 11.427, 'eval_steps_per_second': 0.073}
14 - {'eval_loss': 1.1978527307510376, 'eval_auc': 0.41633144105220765, 'eval_accuracy': 0.21860124496521421, 'eval_precision_macro': 0.21315751971043853, 'eval_recall_macro': 0.2797625003370121, 'eval_runtime': 712.1215, 'eval_samples_per_second': 11.505, 'eval_steps_per_second': 0.073}
15 - {'eval_loss': 1.1950807571411133, 'eval_auc': 0.44866085801099714, 'eval_accuracy': 0.23056267545465642, 'eval_precision_macro': 0.15542522671481704, 'eval_recall_macro': 0.2877277775425127, 'eval_runtime': 718.0506, 'eval_samples_per_second': 11.41, 'eval_steps_per_second': 0.072}
16 - {'eval_loss': 1.1978527307510376, 'eval_auc': 0.41633144105220765, 'eval_accuracy': 0.21860124496521421, 'eval_precision_macro': 0.21315751971043853, 'eval_recall_macro': 0.2797625003370121, 'eval_runtime': 719.3546, 'eval_samples_per_second': 11.389, 'eval_steps_per_second': 0.072}
17 - {'eval_loss': 1.1978527307510376, 'eval_auc': 0.41633144105220765, 'eval_accuracy': 0.21860124496521421, 'eval_precision_macro': 0.21315751971043853, 'eval_recall_macro': 0.2797625003370121, 'eval_runtime': 714.1746, 'eval_samples_per_second': 11.472, 'eval_steps_per_second': 0.073}
18 - 



"""