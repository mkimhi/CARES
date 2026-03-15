#!/usr/bin/env python3
"""
Compute CARES sufficient resolution per (image, question) and write new parquet copies.

Output columns (per row):
  - cares_predicted_res: list[float]   # predicted sufficient max-side (expected value)
  - cares_sufficient_res: list[int]    # min(predicted, original_max_side) per image
"""

import os
import re
import sys
import math
import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

import pyarrow as pa
import pyarrow.parquet as pq

from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel, PeftConfig

# Your mapping (class 0/1/2 -> resolution)
RESOLUTION_MAP = {0: 384, 1: 768, 2: 1024}


def get_rank_info() -> Tuple[int, int, int]:
    """Best-effort rank/world/local_rank detection for torchrun / MPI / SLURM-ish envs."""
    def _get_int(*keys: str, default: int) -> int:
        for k in keys:
            v = os.getenv(k)
            if v is not None and v != "":
                try:
                    return int(v)
                except ValueError:
                    pass
        return default

    rank = _get_int("RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID", default=0)
    world = _get_int("WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "SLURM_NTASKS", default=1)
    local_rank = _get_int("LOCAL_RANK", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID", default=0)
    return rank, world, local_rank


def load_and_resize_image(path: str, max_size: Optional[float] = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if max_size is None:
        return img
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    s = float(max_size) / float(max(w, h))
    return img.resize((round(w * s), round(h * s)), Image.BICUBIC)


def token_id_for_digit(tokenizer, digit: str) -> int:
    ids = tokenizer.encode(digit, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Could not encode digit {digit!r}")
    return ids[-1]


class GraniteDoclingGateHF:
    def __init__(
        self,
        adapter_repo: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        token: Optional[str] = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        peft_cfg = PeftConfig.from_pretrained(adapter_repo, token=token)
        base_model_name = peft_cfg.base_model_name_or_path

        self.processor = AutoProcessor.from_pretrained(adapter_repo)

        if torch_dtype is None:
            torch_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )

        base_model = AutoModelForVision2Seq.from_pretrained(
            base_model_name, torch_dtype=torch_dtype
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_repo)
        self.model.to(self.device).eval()

        tok = self.processor.tokenizer
        self.class_token_ids = [
            token_id_for_digit(tok, "0"),
            token_id_for_digit(tok, "1"),
            token_id_for_digit(tok, "2"),
        ]

    @torch.no_grad()
    def predict_batch(self, images: List[Image.Image], questions: List[str]) -> List[float]:
        """
        Batch version: returns expected resolution (float) per sample.
        """
        assert len(images) == len(questions) and len(images) > 0

        messages_batch = []
        for q in questions:
            messages_batch.append(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
            )

        prompts = [
            self.processor.apply_chat_template(m, add_generation_prompt=True)
            for m in messages_batch
        ]

        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)
        #next_token_logits = outputs.logits[:, -1, :]
        attn = inputs.get("attention_mask", None)
        if attn is None:
            # fallback: assume no padding differences
            idx = torch.full((outputs.logits.size(0),), outputs.logits.size(1)-1, device=outputs.logits.device)
        else:
            # index of last real token for each sample
            idx = attn.long().sum(dim=1) - 1  # [B]

        batch_idx = torch.arange(outputs.logits.size(0), device=outputs.logits.device)
        next_token_logits = outputs.logits[batch_idx, idx, :]  # [B, vocab]
        class_logits = next_token_logits[:, self.class_token_ids]
        probs = F.softmax(class_logits, dim=-1).detach().float().cpu()  # [B,3]

        expected = []
        for p in probs:
            exp_res = float(sum(RESOLUTION_MAP[i] * float(p[i]) for i in range(3)))
            expected.append(exp_res)
        return expected


def extract_question(messages: Any, conversations: Any) -> str:
    """
    Tries to pull the user's text question out of either:
      - messages: list of dicts with ["content"] list containing {"text": "..."}
      - conversations: list of dicts with {"from":"human","value":"..."}
    """
    text = ""

    # Prefer messages
    if isinstance(messages, list) and messages:
        # find first user-like message
        msg0 = messages[0]
        if isinstance(msg0, dict):
            content = msg0.get("content")
            if isinstance(content, list):
                parts = []
                for c in content:
                    if isinstance(c, dict) and "text" in c and isinstance(c["text"], str):
                        parts.append(c["text"])
                if parts:
                    text = "\n".join(parts)

    # Fallback to conversations
    if not text and isinstance(conversations, list):
        for turn in conversations:
            if isinstance(turn, dict) and turn.get("from") == "human" and isinstance(turn.get("value"), str):
                text = turn["value"]
                break

    # Clean common image placeholders
    text = re.sub(r"<image>\s*", "", text)
    text = text.strip()
    return text


def extract_image_paths(image_field: Any) -> List[str]:
    """
    image column looks like: [{'bytes': None, 'path': '/...'}, ...]
    """
    paths: List[str] = []
    if isinstance(image_field, list):
        for it in image_field:
            if isinstance(it, dict) and isinstance(it.get("path"), str):
                paths.append(it["path"])
            elif isinstance(it, str):
                paths.append(it)
    return paths


def orig_max_side(image_sizes_field: Any, idx: int) -> Optional[int]:
    """
    image_sizes looks like [[w,h],[w,h],...]
    """
    if not isinstance(image_sizes_field, list):
        return None
    if idx >= len(image_sizes_field):
        return None
    sz = image_sizes_field[idx]
    if isinstance(sz, (list, tuple)) and len(sz) >= 2:
        try:
            return int(max(int(sz[0]), int(sz[1])))
        except Exception:
            return None
    return None


def process_parquet(
    in_path: Path,
    out_path: Path,
    gate: GraniteDoclingGateHF,
    probe_size: int,
    arrow_batch_size: int,
    infer_batch_size: int,
    compression: str,
    overwrite: bool,
    col_prefix: str,
    log_every_batches: int = 10,
) -> Tuple[int, int]:
    """
    Returns (rows_processed, images_processed).
    """
    if out_path.exists() and not overwrite:
        return (0, 0)

    pf = pq.ParquetFile(str(in_path))
    writer: Optional[pq.ParquetWriter] = None

    rows_done = 0
    imgs_done = 0
    batch_i = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    for batch in pf.iter_batches(batch_size=arrow_batch_size):
        table = pa.Table.from_batches([batch])

        cols = set(table.schema.names)
        img_col = table["image"].to_pylist() if "image" in cols else [None] * table.num_rows
        msg_col = table["messages"].to_pylist() if "messages" in cols else [None] * table.num_rows
        conv_col = table["conversations"].to_pylist() if "conversations" in cols else [None] * table.num_rows
        sz_col = table["image_sizes"].to_pylist() if "image_sizes" in cols else [None] * table.num_rows

        pred_per_row: List[List[Optional[float]]] = []
        suff_per_row: List[List[Optional[int]]] = []

        # build per-row scaffolding
        row_img_paths: List[List[str]] = []
        row_questions: List[str] = []

        for r in range(table.num_rows):
            q = extract_question(msg_col[r], conv_col[r])
            paths = extract_image_paths(img_col[r])
            row_questions.append(q)
            row_img_paths.append(paths)
            pred_per_row.append([None] * len(paths))
            suff_per_row.append([None] * len(paths))

        # task buffers for batched gate inference
        buf_imgs: List[Image.Image] = []
        buf_qs: List[str] = []
        buf_ptrs: List[Tuple[int, int, Optional[int]]] = []  # (row_idx, img_idx, orig_max)

        def flush():
            nonlocal imgs_done
            if not buf_imgs:
                return
            preds = gate.predict_batch(buf_imgs, buf_qs)
            for (row_idx, img_idx, omax), p in zip(buf_ptrs, preds):
                pred_per_row[row_idx][img_idx] = float(p)
                if omax is not None and omax < p:
                    chosen = int(omax)         # keep original if it's smaller than predicted
                else:
                    chosen = int(round(p))     # otherwise use predicted (rounded)
                suff_per_row[row_idx][img_idx] = chosen
            imgs_done += len(buf_imgs)
            buf_imgs.clear()
            buf_qs.clear()
            buf_ptrs.clear()

        # fill tasks
        for r in range(table.num_rows):
            q = row_questions[r]
            paths = row_img_paths[r]
            for j, pth in enumerate(paths):
                omax = orig_max_side(sz_col[r], j)
                try:
                    probe_img = load_and_resize_image(pth, probe_size)
                    buf_imgs.append(probe_img)
                    buf_qs.append(q)
                    buf_ptrs.append((r, j, omax))
                except Exception as e:
                    # If we can't load, leave predicted null; keep original max side if exists.
                    pred_per_row[r][j] = None
                    suff_per_row[r][j] = int(omax) if omax is not None else None

                if len(buf_imgs) >= infer_batch_size:
                    flush()

        flush()

        pred_arr = pa.array(
            pred_per_row,
            type=pa.list_(pa.float32()),
        )
        suff_arr = pa.array(
            suff_per_row,
            type=pa.list_(pa.int32()),
        )

        pred_name = f"{col_prefix}_predicted_res"
        suff_name = f"{col_prefix}_sufficient_res"

        # If columns already exist, drop them first to avoid schema mismatch
        if pred_name in cols:
            table = table.drop([pred_name])
        if suff_name in cols:
            table = table.drop([suff_name])

        table = table.append_column(pred_name, pred_arr)
        table = table.append_column(suff_name, suff_arr)

        if writer is None:
            writer = pq.ParquetWriter(
                where=str(out_path),
                schema=table.schema,
                compression=compression,
                use_dictionary=True,
            )
        writer.write_table(table)

        rows_done += table.num_rows
        batch_i += 1
        if (batch_i % log_every_batches) == 0:
            print(f"[{in_path.name}] batches={batch_i} rows={rows_done} imgs={imgs_done}", flush=True)

    if writer is not None:
        writer.close()

    return rows_done, imgs_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", type=str, required=True)
    ap.add_argument("--output_root", type=str, required=True)
    ap.add_argument("--adapter_repo", type=str, default="Kimhi/granite-docling-res-gate-lora")
    ap.add_argument("--probe_size", type=int, default=256)
    ap.add_argument("--arrow_batch_size", type=int, default=64)
    ap.add_argument("--infer_batch_size", type=int, default=16)
    ap.add_argument("--compression", type=str, default="zstd", choices=["zstd", "snappy", "gzip", "brotli", "lz4", "none"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--col_prefix", type=str, default="cares")
    ap.add_argument("--max_files", type=int, default=-1)
    ap.add_argument("--fail_fast", action="store_true")
    args = ap.parse_args()

    rank, world, local_rank = get_rank_info()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        # small speed knobs that are usually safe
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    in_root = Path(args.input_root)
    out_root = Path(args.output_root)

    all_files = sorted([p for p in in_root.rglob("*.parquet") if p.is_file()])
    if args.max_files and args.max_files > 0:
        all_files = all_files[: args.max_files]





    #split files among ranks...
    my_files = all_files[rank::world]

    print(f"[rank {rank}/{world} local_rank {local_rank}] files={len(my_files)} / total={len(all_files)}", flush=True)

    gate = GraniteDoclingGateHF(adapter_repo=args.adapter_repo)

    total_rows = 0
    total_imgs = 0

    for i, fpath in enumerate(my_files):
        rel = fpath.relative_to(in_root)
        out_path = out_root / rel
        try:
            rows, imgs = process_parquet(
                in_path=fpath,
                out_path=out_path,
                gate=gate,
                probe_size=args.probe_size,
                arrow_batch_size=args.arrow_batch_size,
                infer_batch_size=args.infer_batch_size,
                compression=(None if args.compression == "none" else args.compression),
                overwrite=args.overwrite,
                col_prefix=args.col_prefix,
            )
            total_rows += rows
            total_imgs += imgs
            print(f"[rank {rank}] done {i+1}/{len(my_files)}: {rel} rows={rows} imgs={imgs}", flush=True)
        except Exception as e:
            print(f"[rank {rank}] ERROR processing {fpath}: {e}", file=sys.stderr, flush=True)
            if args.fail_fast:
                raise

    print(f"[rank {rank}] FINISHED rows={total_rows} imgs={total_imgs}", flush=True)


if __name__ == "__main__":
    main()
