##python tg.py --out gate_lr1-3b128 --lr 1e-3 --bsz 128 --resume

#python tg2.py --parquet hardness_data_mix.parquet --model_name google/siglip-so400m-patch14-384 --text_encoder_name sentence-transformers/all-mpnet-base-v2 --text_pooling mean --gate_type attn --out './gate_text_encoder'

##binary
#python tg.py --out './tmp2' --resume
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
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None 
SAFE_MAX_PIXELS = 150_000_000 
ImageFile.LOAD_TRUNCATED_IMAGES = True # optional
import math
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

BINARY=False#True#False

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
                if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.webp'))]
        if not imgs:
            raise FileNotFoundError(f"No image files in {path}")
        path = os.path.join(path, imgs[0])

    # Open lazily (no full decode yet)
    im = Image.open(path)
    #w, h = im.size
    #pixels = w * h

    # if pixels > SAFE_MAX_PIXELS:
    #     # compute integer reduction so (w/r)*(h/r) <= SAFE_MAX_PIXELS
    #     r = max(2, math.ceil((pixels / SAFE_MAX_PIXELS) ** 0.5))
    #     try:
    #         # JPEG/PNG/WebP support 'reduce' to shrink *before* decode
    #         im.close()
    #         im = Image.open(path, reduce=r)
    #     except TypeError:
    #         # Fallback: allow load, then downscale (may be heavier)
    #         im.load()
    #         im.thumbnail((w // r, h // r), Image.Resampling.LANCZOS)

    return im.convert("RGB")

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

        if BINARY:
            labels = [int(f['labels'][0]) for f in features]
        else:
            #print(f['labels'] for f in features)
            labels = [int(f['labels'][0]) for f in features]
        
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
        if BINARY:
            labels = torch.tensor(labels, dtype=torch.float)
        else:
            labels = torch.tensor(labels, dtype=torch.long)
        return {
            'pixel_values': pixel_values,
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(padded_attention_masks, dtype=torch.long),
            'labels': labels#torch.stack(labels)
        }


class HardnessDS(Dataset):
    def __init__(
        self, parquet_path, model_name,
        split='train', val_frac=0.1, random_state=42
    ):
        df = pd.read_parquet(parquet_path)
        train_df, val_df = train_test_split(
            df, test_size=val_frac, stratify=df['hard'], random_state=random_state
        )
        self.df = train_df if split == 'train' else val_df
        
        # Use the appropriate processor for the model
        if "granite" in model_name.lower():
            self.proc = AutoProcessor.from_pretrained(model_name)
            self.tokenizer = self.proc.tokenizer
            self.image_proc = SiglipProcessor.from_pretrained('google/siglip-so400m-patch14-384')
        elif "siglip" in model_name.lower():
            self.proc = SiglipProcessor.from_pretrained(model_name)
            self.tokenizer = self.proc.tokenizer
            self.image_proc = self.proc
        elif "Qwen" in model_name or "mistral" in model_name:
            self.proc = AutoProcessor.from_pretrained(model_name)
            self.tokenizer = self.proc.tokenizer
            self.image_proc = self.prochttps://ca.slack-edge.com/E27SFGS2W-U08MQKD3DNU-25921df0e7e7-512
        else:
            self.proc = CLIPProcessor.from_pretrained(model_name)
            self.tokenizer = self.proc.tokenizer
            self.image_proc = self.proc
            
        # Get the actual max length from tokenizer or model config
        if hasattr(self.tokenizer, 'model_max_length') and self.tokenizer.model_max_length < 1000000:
            self.max_length = self.tokenizer.model_max_length
        else:
            print('set fixed max_length=8192')
            self.max_length = 8192
        
        #     # Fallback based on model type
        #     if "siglip" in model_name.lower() or "granite" in model_name.lower():
        #         self.max_length = 64  # SigLIP typically has 64 max position embeddings
        #     else:
        #         self.max_length = 77  # CLIP typically has 77
            
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # For some models, we might need to add a pad token
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = safe_load_image(row.high_path) #low_path high_path
        
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
        if BINARY:
            labels = torch.tensor(1 if int(row.hard) > 0 else 0, dtype=torch.long),
        else:
            labels = torch.tensor(int(row.hard), dtype=torch.long),  # 0,1,2
        return {
            'pixel_values': img_inputs['pixel_values'].squeeze(0),
            'input_ids': text_inputs['input_ids'].squeeze(0).tolist(),
            'attention_mask': text_inputs['attention_mask'].squeeze(0).tolist(),
            'labels': labels,  # 0,1,2
        }


class MeanPoolGate(nn.Module):
    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        if BINARY:
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, 3)
            )

    def forward(self, feats_img, feats_txt=None):
        feats = torch.cat([feats_img, feats_txt], dim=1)
        return self.net(feats.mean(dim=1)).squeeze(-1)


class AttnGate(nn.Module):
    def __init__(self, dim, heads=4, hidden_dim=256): 
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        if not BINARY:
            self.mlp = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, 3)
            )
        else:
            self.mlp = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, feats_img, feats_txt):
        #feats = torch.cat([feats_img, feats_txt], dim=1)
        #B = feats.size(0)
        #D = feats.size(-1)
        #cls = self.cls_token.expand(B, 1, D)
        pooled, _ = self.attn(query=feats_txt, key=feats_img, value=feats_img, need_weights=False)
        #pooled = pooled.mean(dim=1) # [B, dim]
        return self.mlp(pooled.squeeze(1)).squeeze(-1)


class AttnGate_dumb(nn.Module):
    def __init__(self, dim, heads=4, hidden_dim=256): 
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim, dtype=torch.float))
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        if not BINARY:
            self.mlp = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, 3)
            )
        else:
            self.mlp = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, feats_img, feats_txt):
        feats = torch.cat([feats_img, feats_txt], dim=1)
        B = feats.size(0)
        D = feats.size(-1)
        cls = self.cls_token.expand(B, 1, D)
        pooled, _ = self.attn(query=cls, key=feats, value=feats, need_weights=False)
        return self.mlp(pooled.squeeze(1)).squeeze(-1)


class GateWrapper(nn.Module):
    def __init__(self, model_name, gate_type='attn'):
        super().__init__()
        self.gate_type = gate_type
        if "granite" in model_name.lower():
            self.clip = SiglipModel.from_pretrained('google/siglip-so400m-patch14-384', torch_dtype=torch.float)
        elif "siglip" in model_name.lower():
            self.clip = SiglipModel.from_pretrained(model_name, torch_dtype=torch.float)
        elif "Qwen" in model_name or "mistral" in model_name:
            self.clip = AutoModel.from_pretrained(model_name, torch_dtype=torch.float)
        else:
            self.clip = CLIPModel.from_pretrained(model_name, torch_dtype=torch.float)
            
        for p in self.clip.parameters():
            p.requires_grad_(False)
        
        # Get the correct dimension from the model config
        if hasattr(self.clip.config, 'vision_config'):
            dim = self.clip.config.vision_config.hidden_size
        elif hasattr(self.clip.config, 'hidden_size'):
            dim = self.clip.config.hidden_size
        else:
            dim = 768  # Default fallback
            
        self.gate = AttnGate(dim) if gate_type == 'attn' else MeanPoolGate(dim)

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        #print('=======================================')
        #print(f'Forward pass with pixel_values shape: {pixel_values.shape}, input_ids shape: {input_ids.shape}')
        #print('=======================================')

        with torch.no_grad():
            out = self.clip(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            if self.gate_type == 'attn':
                # Handle different output formats
                if hasattr(out, 'vision_model_output') and hasattr(out, 'text_model_output'):
                    img_feats = out.vision_model_output.last_hidden_state #b x Seq_img x d
                    txt_feats = out.text_model_output.last_hidden_state[:,-1,:].unsqueeze(1) #b x Seq_txt x d
                elif hasattr(out, 'image_embeds') and hasattr(out, 'text_embeds'):
                    img_feats = out.image_embeds.unsqueeze(1)
                    txt_feats = out.text_embeds.unsqueeze(1)
                else:
                    # Fallback for models with different output structure
                    img_feats = out.vision_model_output.last_hidden_state[:, 0:1, :]  # CLS token
                    txt_feats = out.text_model_output.last_hidden_state[:, 0:1, :]   # CLS token
            else:
                img_feats = out.image_embeds.unsqueeze(1)
                txt_feats = out.text_embeds.unsqueeze(1)
        logits = self.gate(img_feats, txt_feats) #bx3
        # logits = (
        #     self.gate(img_feats, txt_feats)
        #     if isinstance(self.gate, AttnGate)
        #     else self.gate(img_feats)
        # )
        
        loss = None
        if labels is not None and BINARY:
             loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        if labels is not None:
            if logits.ndim == 2 and logits.size(-1) > 1:      # multi-class (e.g., 3)
                loss = nn.functional.cross_entropy(logits, labels)
            else:                                             # binary head
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, labels.float()
                    ) #logits.squeeze(-1)
        
        # #print('=======================================')
        #print(f'Logits: {logits}, Labels shape: {labels.shape if labels is not None else "N/A"}')
        #print('=======================================')
        return {'loss': loss, 'logits': logits}



def compute_metrics(eval_pred):
    if BINARY:
        return compute_metrics_2(eval_pred)
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
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
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

def compute_metrics_debug(eval_pred):
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
    parser.add_argument('--gate_type',   choices=['mean','attn'], default='attn')
    parser.add_argument('--out',         default='./gate_ckpt')
    parser.add_argument('--bsz',         type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--epochs',      type=int,   default=6)
    parser.add_argument('--val_frac',    type=float, default=0.1)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--eval_only',     type=bool,   default=False)
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
        'val', args.val_frac, random_state=args.seed
    )
    
    model = GateWrapper(args.model_name, gate_type=args.gate_type)
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
            'train', args.val_frac, random_state=args.seed
        )
        val_ds = HardnessDS(
            args.parquet, args.model_name,
            'val', args.val_frac, random_state=args.seed
        )
        
        model = GateWrapper(args.model_name, gate_type=args.gate_type)
        
        # Use custom data collator
        data_collator = CustomDataCollator(tokenizer=train_ds.tokenizer)
        if BINARY:
            training_args = TrainingArguments(
                output_dir=args.out,
                per_device_train_batch_size=args.bsz,
                per_device_eval_batch_size=args.bsz,
                num_train_epochs=args.epochs,
                learning_rate=args.lr,
                bf16=True,
                logging_steps=20,
                eval_strategy='steps', #
                eval_steps=100,        # evaluate every 100 steps
                save_safetensors=True, #for resume!!
                save_strategy='steps',
                save_steps=100,        # save every 100 steps
                load_best_model_at_end=True,
                metric_for_best_model='auc',
                greater_is_better=True,  # AUC higher is better
            )
        else:
            training_args = TrainingArguments(
                output_dir=args.out,
                per_device_train_batch_size=args.bsz,
                per_device_eval_batch_size=args.bsz,
                num_train_epochs=args.epochs,
                learning_rate=args.lr,                 # try 5e-4 to 1e-3 for a tiny head
                bf16=True,
                logging_steps=20,
                eval_strategy='steps',
                eval_steps = 100,        # evaluate every 100 steps
                save_steps=100,          # save every 100 steps
                save_strategy='steps',
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