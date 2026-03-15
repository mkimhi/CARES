import argparse, os, re, json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from datasets import DatasetDict, Dataset as HFDataset

from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from transformers.trainer_utils import get_last_checkpoint

# --- PIL safety for large / truncated images
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ------------------------ IO utils ------------------------
def load_image(path: str, target_size=None) -> Image.Image:
    """Robust loader: accepts a directory (picks first image) and fixes '_<num>.ext' suffixes."""
    if os.path.isdir(path):
        imgs = [
            f
            for f in os.listdir(path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
        ]
        if not imgs:
            raise FileNotFoundError(f"No image files in {path}")
        path = os.path.join(path, imgs[0])

    try:
        im = Image.open(path).convert("RGB")
    except FileNotFoundError:
        alt_path = re.sub(
            r"_(\d+)(?=\.(jpg|jpeg|png|bmp|webp)$)", "", path, flags=re.IGNORECASE
        )
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


def bf16_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


# ------------------------ Dataset (as HF Dataset) ------------------------
def build_hf_datasets(
    parquet_path: str,
    val_frac: float,
    seed: int,
    binary: bool,
) -> DatasetDict:
    df = pd.read_parquet(parquet_path)
    df = df[["question", "mid_path", "hard"]].copy()

    if binary:
        df["label"] = (df["hard"].astype(int) > 0).astype(int)
    else:
        df["label"] = df["hard"].astype(int)

    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(
        df,
        test_size=val_frac,
        stratify=df["hard"],
        random_state=seed,
    )

    train_ds = HFDataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = HFDataset.from_pandas(val_df.reset_index(drop=True))

    return DatasetDict(train=train_ds, validation=val_ds)


# ------------------------ Collator ------------------------
class DoclingSFTCollator:
    """
    Collator that:
    - builds chat-style prompts: image + question -> assistant label "0"/"1"/"2"
    - uses AutoProcessor.apply_chat_template
    - masks pad and image tokens in the labels (others are standard LM loss)
    """

    def __init__(self, processor, target_size: Tuple[int, int] | None = None):
        self.processor = processor
        self.tokenizer = processor.tokenizer

        # infer default image size from processor if not provided
        if target_size is None and hasattr(processor, "image_processor"):
            size_cfg = getattr(processor.image_processor, "size", None)
            edge = None
            if isinstance(size_cfg, dict):
                edge = (
                    size_cfg.get("longest_edge")
                    or size_cfg.get("shortest_edge")
                    or size_cfg.get("height")
                    or size_cfg.get("width")
                )
            if edge is not None:
                target_size = (edge, edge)
        if target_size is None:
            target_size = (512, 512)  # safe default for idefics3-ish models

        self.target_size = target_size

        image_tok = getattr(self.processor, "image_token", "<image>")
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(image_tok)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts: List[str] = []
        images: List[Image.Image] = []

        for ex in examples:
            question = str(ex["question"])
            label_int = int(ex["label"])
            label_str = str(label_int)

            img = load_image(ex["mid_path"], target_size=self.target_size)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": label_str}],
                },
            ]

            chat = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(chat)
            images.append(img)

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        labels[labels == pad_id] = -100
        if self.image_token_id is not None and self.image_token_id != -1:
            labels[labels == self.image_token_id] = -100

        batch["labels"] = labels
        return batch


# ------------------------ Trainable set-up ------------------------
def configure_trainable_parts(
    model,
    llm_lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    train_last_proj: bool,
):
    # 1) Freeze all base params
    for p in model.parameters():
        p.requires_grad = False

    # 2) LoRA on text projections (if requested)
    if llm_lora_r > 0:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

        peft_config = LoraConfig(
            r=llm_lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules,   # <-- changed line
            task_type="CAUSAL_LM",
            use_dora=False,
            init_lora_weights="gaussian",
        )

        #model.enable_input_require_grads() #return if want to use gradient checkpointing!
        model = get_peft_model(model, peft_config)

    # 3) Last projection / LM head: fully trainable if requested
    if train_last_proj:
        if hasattr(model, "lm_head"):
            for p in model.lm_head.parameters():
                p.requires_grad = True
        else:
            for name, p in model.named_parameters():
                if "lm_head" in name:
                    p.requires_grad = True

    # Debug: print trainable params
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    print(
        f"Trainable params: {trainable:,} / {total:,} "
        f"({100.0 * trainable / max(total, 1):.4f}%)"
    )
    return model

# ------------------------ CLI / Train ------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument(
        "--model_name",
        default="ibm-granite/granite-docling-258M",
        help="Idefics3 VLM (e.g. Granite-Docling or SmolVLM)",
    )
    ap.add_argument("--out", default="./docling_sft_gate")
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)#5e-5)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--binary", action="store_true")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume_from", type=str, default=None)
    # ---- NEW: granularity knobs ----
    ap.add_argument(
        "--llm_lora_r",
        type=int,
        default=0,
        help="If >0, apply LoRA of this rank on language model projections (e.g. 4 or 8).",
    )
    ap.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha (scale).",
    )
    ap.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout.",
    )
    ap.add_argument(
        "--train_last_proj",
        action="store_true",
        help="If set, fully train the last projection / lm_head.",
    )

    return ap.parse_args()


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ----- build HF datasets -----
    dsets = build_hf_datasets(args.parquet, args.val_frac, args.seed, args.binary)

    # ----- processor -----
    processor = AutoProcessor.from_pretrained(args.model_name)

    collator = DoclingSFTCollator(processor)

    # ----- model -----
    model_dtype = torch.bfloat16 if bf16_available() else torch.float16
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        dtype=model_dtype,
    )

    # ----- configure what is trainable -----
    model = configure_trainable_parts(
        model=model,
        llm_lora_r=args.llm_lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        train_last_proj=args.train_last_proj,
    )

        # ----- SFT config (TRL) -----
    training_args = SFTConfig(
        output_dir=args.out,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,        # keep this, SFTConfig usually accepts it
        save_total_limit=2,
        bf16=bf16_available(),
        fp16=not bf16_available(),
        gradient_checkpointing=False,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",             # we own the collator
        dataset_kwargs={"skip_prepare_dataset": True},
        #max_steps=args.max_steps,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dsets["train"],
        eval_dataset=dsets["validation"],
        data_collator=collator,
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
