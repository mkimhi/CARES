"""
Train the HardnessGate (mean-pool or attention) that decides whether to
run Granite-Vision on native or high-res tokens.
Updated script includes:
- Robust train/validation split with sklearn's train_test_split
- Attention mask handling for text inputs
- Custom data collator for handling both images and text
- Correct metric key and numpy conversion
- Option to adjust validation split fraction
- Added pad_token to tokenizer when missing
"""

import argparse
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, roc_curve, precision_recall_curve
)
from sklearn.model_selection import train_test_split


from torch import nn
from torch.utils.data import Dataset
from transformers import (
    SiglipModel, SiglipProcessor, AutoTokenizer, AutoProcessor,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    CLIPModel, CLIPProcessor,AutoModel)
from transformers.trainer_utils import get_last_checkpoint

from safetensors.torch import load_file



def safe_load_image(path):
    if os.path.exists(path):
        return load_image(path)
    else:
        # try removing the trailing "_<digit>" before extension
        base, ext = os.path.splitext(path)
        if "_" in base and base.split("_")[-1].isdigit():
            alt_path = "_".join(base.split("_")[:-1]) + ext
            if os.path.exists(alt_path):
                return load_image(alt_path)
    return None  # or raise an error if neither exists



def load_image(path):
    if os.path.isdir(path):
        imgs = [f for f in os.listdir(path)
                if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]
        if not imgs:
            raise FileNotFoundError(f"No image files in {path}")
        path = os.path.join(path, imgs[0])
    return Image.open(path).convert('RGB')


@dataclass
class CustomDataCollator:
    """Custom data collator for handling both images and text"""
    tokenizer: Any
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Separate the different types of data
        pixel_values = [f['pixel_values'] for f in features]
        input_ids = [f['input_ids'] for f in features]
        attention_masks = [f['attention_mask'] for f in features]


        #input_ids  = input_ids[:64]  # Limit to 64 for performance
        #attention_masks = attention_masks[:64]

        labels = [int(f['labels']) for f in features]

        
        # Stack pixel values (they should all have the same shape)
        pixel_values = torch.stack(pixel_values)
        
        # Pad text inputs
        max_length = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        padded_attention_masks = []
        
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        
        for ids, mask in zip(input_ids, attention_masks):
            padding_length = max_length - len(ids)
            padded_ids = ids + [pad_token_id] * padding_length
            padded_mask = mask + [0] * padding_length
            padded_input_ids.append(padded_ids)
            padded_attention_masks.append(padded_mask)
        
        return {
            'pixel_values': pixel_values,
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(padded_attention_masks, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)#torch.stack(labels)
        }


class HardnessDS(Dataset):
    def __init__(self, parquet_path, model_name, split='train', val_frac=0.1, random_state=42,
                 text_encoder_name=None):
        df = pd.read_parquet(parquet_path)
        train_df, val_df = train_test_split(
            df, test_size=val_frac, stratify=df['hard'], random_state=random_state
        )
        self.df = train_df if split == 'train' else val_df

        # image processor follows the vision tower choice
        if "granite" in model_name.lower():
            self.proc = AutoProcessor.from_pretrained(model_name)
            self.image_proc = SiglipProcessor.from_pretrained('google/siglip-so400m-patch14-384')
        elif "siglip" in model_name.lower():
            self.proc = SiglipProcessor.from_pretrained(model_name)
            self.image_proc = self.proc
        else:
            self.proc = CLIPProcessor.from_pretrained(model_name)
            self.image_proc = self.proc

        # tokenizer: prefer external long-context encoder if provided
        if text_encoder_name:
            self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name, use_fast=True)
        else:
            self.tokenizer = self.proc.tokenizer

        # max length from tokenizer (fallback to 4096)
        self.max_length = (
            self.tokenizer.model_max_length if getattr(self.tokenizer, 'model_max_length', 0) and
            self.tokenizer.model_max_length < 1000000 else 4096
        )

        # ensure pad token
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = safe_load_image(row.high_path) #low_path
        
        # Process image
        img_inputs = self.image_proc(images=img, return_tensors='pt')
        
        # Process text using tokenizer directly for better control
        text_inputs = self.tokenizer(
            row.question, 
            return_tensors='pt',
            padding=False, 
            truncation=True, 
            max_length=self.max_length,
            return_attention_mask=True
        )
        
        return {
            'pixel_values': img_inputs['pixel_values'].squeeze(0),
            'input_ids': text_inputs['input_ids'].squeeze(0).tolist(),
            'attention_mask': text_inputs['attention_mask'].squeeze(0).tolist(),
            'labels': torch.tensor(int(row.hard), dtype=torch.long),  # 0,1,2
        }


class MeanPoolGate(nn.Module):
    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feats_img, feats_txt=None):
        return self.net(feats_img.mean(dim=1)).squeeze(-1)


class AttnGate(nn.Module):
    def __init__(self, dim, heads=4, hidden_dim=256,three=True): #todo:fix via args!!!
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim, dtype=torch.float))
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, feats_img, feats_txt):
        feats = torch.cat([feats_img, feats_txt], dim=1)
        B = feats.size(0)
        D = feats.size(-1)
        cls = self.cls_token.expand(B, 1, D)
        pooled, _ = self.attn(query=cls, key=feats, value=feats, need_weights=False)
        return self.mlp(pooled.squeeze(1)).squeeze(-1)

class GateWrapper(nn.Module):
    def __init__(self, model_name, gate_type='attn', text_encoder_name=None, text_pooling='cls'):
        super().__init__()
        # Vision tower: same as before
        if "granite" in model_name.lower():
            self.clip = SiglipModel.from_pretrained('google/siglip-so400m-patch14-384', torch_dtype=torch.float)
        elif "siglip" in model_name.lower():
            self.clip = SiglipModel.from_pretrained(model_name, torch_dtype=torch.float)
        else:
            self.clip = CLIPModel.from_pretrained(model_name, torch_dtype=torch.float)
        
        for p in self.clip.parameters():
            p.requires_grad_(False)

        # vision hidden size
        if hasattr(self.clip.config, 'vision_config'):
            dim = self.clip.config.vision_config.hidden_size
        elif hasattr(self.clip.config, 'hidden_size'):
            dim = self.clip.config.hidden_size
        else:
            dim = 768

        # Text encoder: external long-context or reuse CLIP/SigLIP text tower
        self.use_external_text = text_encoder_name is not None
        self.text_pooling = text_pooling

        if self.use_external_text:
            self.text_enc = AutoModel.from_pretrained(text_encoder_name, torch_dtype=torch.float)
            for p in self.text_enc.parameters():
                p.requires_grad_(False)
            tdim = getattr(self.text_enc.config, 'hidden_size', dim)
            # project pooled text to vision dim
            self.text_proj = nn.Linear(tdim, dim, bias=False)

        else:
            self.text_enc = None  # will reuse CLIP/SigLIP text pathway

        self.gate = AttnGate(dim) if gate_type == 'attn' else MeanPoolGate(dim)

    def _pool_text(self, last_hidden_state, attention_mask):
        if self.text_pooling == 'cls':
            return last_hidden_state[:, 0]  # [B, H]
        # mean pool over valid tokens
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # [B, T, 1]
        summed = (last_hidden_state * mask).sum(dim=1)                   # [B, H]
        denom = mask.sum(dim=1).clamp(min=1e-6)                          # [B, 1]
        return summed / denom

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        # --- VISION: use vision submodule only ---
        with torch.no_grad():
            if isinstance(self.clip, SiglipModel):
                vout = self.clip.vision_model(pixel_values=pixel_values, return_dict=True)
                img_feats = vout.last_hidden_state                      # [B, Tv, D]
            elif isinstance(self.clip, CLIPModel):
                vout = self.clip.vision_model(pixel_values=pixel_values, return_dict=True)
                img_feats = vout.last_hidden_state                      # [B, Tv, D]
            else:
                # Fallback: try to access a vision_model-like output
                vout = self.clip.vision_model(pixel_values=pixel_values, return_dict=True)
                img_feats = getattr(vout, "last_hidden_state", vout[0])

            if img_feats.dim() == 2:
                img_feats = img_feats.unsqueeze(1)  # [B,1,D] safety

            # --- TEXT: external encoder (recommended) or fallback to model text submodule ---
            if self.use_external_text:
                tout = self.text_enc(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                tpooled = self._pool_text(tout.last_hidden_state, attention_mask)  # [B, Ht]
                txt_feats = self.text_proj(tpooled).unsqueeze(1)                   # [B,1,D]
            else:
                if isinstance(self.clip, SiglipModel):
                    tout = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                    # use CLS (token 0) or mean—CLS is standard for SigLIP/CLIP
                    tpooled = tout.last_hidden_state[:, 0]                         # [B, Ht]
                    # project to vision dim if sizes differ
                    if not hasattr(self, "text_proj_fallback"):
                        tdim = tout.last_hidden_state.size(-1)
                        vdim = img_feats.size(-1)
                        self.text_proj_fallback = nn.Linear(tdim, vdim, bias=False).to(img_feats.device)
                    txt_feats = self.text_proj_fallback(tpooled).unsqueeze(1)
                elif isinstance(self.clip, CLIPModel):
                    tout = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                    tpooled = tout.last_hidden_state[:, 0]
                    if not hasattr(self, "text_proj_fallback"):
                        tdim = tout.last_hidden_state.size(-1)
                        vdim = img_feats.size(-1)
                        self.text_proj_fallback = nn.Linear(tdim, vdim, bias=False).to(img_feats.device)
                    txt_feats = self.text_proj_fallback(tpooled).unsqueeze(1)
                else:
                    raise RuntimeError("No valid text path found")

        # --- GATE ---
        logits = self.gate(img_feats, txt_feats) if isinstance(self.gate, AttnGate) else self.gate(img_feats)

        loss = None
        if labels is not None:
            if logits.ndim == 2 and logits.size(-1) > 1:
                loss = nn.functional.cross_entropy(logits, labels)
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float())

        return {'loss': loss, 'logits': logits}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()  # [N,3]
    preds = probs.argmax(axis=-1)

    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average='macro', zero_division=0)
    rec  = recall_score(labels, preds, average='macro', zero_division=0)

    # Optional: multiclass AUC (needs prob per class)
    try:
        auc = roc_auc_score(labels, probs, multi_class='ovr')
    except ValueError:
        auc = float('nan')

    return {'auc': auc, 'accuracy': acc, 'precision_macro': prec, 'recall_macro': rec}

def compute_metrics_2(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.from_numpy(logits)).numpy()
    labels = np.array(labels)

    preds = (probs > 0.5).astype(int)
    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)

    fpr, tpr, roc_thresh = roc_curve(labels, probs)

    # Save ROC plot
    os.makedirs("metrics_debug", exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("metrics_debug/roc_curve.png")
    plt.close()

    return {
        'auc': auc,
        'accuracy': acc,
        'precision@0.5': prec,
        'recall@0.5': rec,
    }



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet',     default='hardness_data_mix.parquet')
    parser.add_argument('--model_name', default='google/siglip-so400m-patch14-384') #ibm-granite/granite-vision-3.3-2b #'granite_vision/models/granite_vision'
    parser.add_argument('--text_encoder_name', type=str, default=None,help='HF name for long-context text encoder (e.g., allenai/longformer-base-4096)')
    parser.add_argument('--text_pooling', choices=['cls','mean'], default='cls',help='How to pool token reps from the text encoder')
    parser.add_argument('--gate_type',   choices=['mean','attn'], default='attn')
    parser.add_argument('--out',         default='./gate_ckpt')
    parser.add_argument('--bsz',         type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--epochs',      type=int,   default=6)
    parser.add_argument('--val_frac',    type=float, default=0.1)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--eval_only',        type=bool,   default=False)
    parser.add_argument('--three',        type=bool,   default=False)
    parser.add_argument('--resume', action='store_true', help='resume from last checkpoint in --out')
    parser.add_argument('--resume_from', type=str, default=None, help='path to a specific checkpoint dir')

    return parser.parse_args()



def load_gate_model(model_name='google/siglip-so400m-patch14-384', gate_type='attn',load_from='./gate_ckpt'):
    gate = GateWrapper(model_name, gate_type=gate_type)
    if load_from is not None:
        state_dict = load_file(os.path.join(load_from, "model.safetensors"))
        gate.load_state_dict(state_dict)
    return gate


def eval(args):
    val_ds = HardnessDS(
        args.parquet, args.model_name,
        'val', args.val_frac, random_state=args.seed, text_encoder_name=args.text_encoder_name
    )
    
    model = GateWrapper(args.model_name, gate_type=args.gate_type,
                        text_encoder_name=args.text_encoder_name,
                        text_pooling=args.text_pooling)
    state_dict = load_file(os.path.join(args.out, "model.safetensors"))
    model.load_state_dict(state_dict)

    data_collator = CustomDataCollator(tokenizer=val_ds.tokenizer)

    training_args = TrainingArguments(
        output_dir=args.out,
        per_device_eval_batch_size=args.bsz,
        do_train=False,
        do_eval=True,
    )




    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )




    metrics = trainer.evaluate()
    print(metrics)

def main():
    args = parse_args()
    if args.eval_only:
        eval(args)
    else:
        train_ds = HardnessDS(
            args.parquet, args.model_name,
            'train', args.val_frac, random_state=args.seed,text_encoder_name=args.text_encoder_name,
        )
        val_ds = HardnessDS(
            args.parquet, args.model_name,
            'val', args.val_frac, random_state=args.seed,text_encoder_name=args.text_encoder_name,
        )
        
        model = GateWrapper(args.model_name, gate_type=args.gate_type,
                        text_encoder_name=args.text_encoder_name,
                        text_pooling=args.text_pooling)
        # Use custom data collator
        data_collator = CustomDataCollator(tokenizer=train_ds.tokenizer)
        
        # training_args = TrainingArguments(
        #     output_dir=args.out,
        #     per_device_train_batch_size=args.bsz,
        #     per_device_eval_batch_size=args.bsz,
        #     num_train_epochs=args.epochs,
        #     learning_rate=args.lr,
        #     bf16=True,
        #     logging_steps=20,
        #     eval_strategy='epoch', #
        #     save_safetensors=True, #for resume!!
        #     save_strategy='epoch',
        #     load_best_model_at_end=True,
        #     metric_for_best_model='auc',
        #     greater_is_better=True,  # AUC higher is better
        # )
        training_args = TrainingArguments(
            output_dir=args.out,
            per_device_train_batch_size=args.bsz,
            per_device_eval_batch_size=args.bsz,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,                 # try 5e-4 to 1e-3 for a tiny head
            bf16=True,
            logging_steps=20,
            eval_strategy='epoch',
            save_strategy='epoch',
            save_safetensors=True, #for resume!!
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',      # AUC can be flaky early in multiclass
            warmup_ratio=0.1,
            weight_decay=0.01,
            lr_scheduler_type='cosine',
            label_smoothing_factor=0.05,
            greater_is_better=True,  # AUC higher is better
            )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        if args.resume or args.resume_from:
            print("Resuming training from checkpoint...")
            resume_ckpt = args.resume_from
            if resume_ckpt is None and args.resume:
                # auto-detect last checkpoint in output_dir
                if os.path.isdir(args.out) and get_last_checkpoint(args.out) is not None:
                    resume_ckpt = get_last_checkpoint(args.out)
            trainer.train(resume_from_checkpoint=resume_ckpt)
        else:
            trainer.train()
        trainer.save_model(args.out)


if __name__ == '__main__':
    main()