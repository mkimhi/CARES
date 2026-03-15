
#!/usr/bin/env python3
"""
Prepare the TextVQA dataset into:
  - images/  (all image files, deduplicated)
  - annotations.jsonl  (one line per Q/A; optionally LLaVA-style "conversations")

Usage:
  python prepare_textvqa.py --out ./textvqa_out --format plain
  python prepare_textvqa.py --out ./textvqa_out_llava --format llava

Requirements:
  pip install datasets pillow tqdm

Notes:
  - Images are saved once per unique OpenImages `image_id` and reused across Q/A pairs.
  - The JSONL contains relative image paths (e.g., "images/abc123.jpg") so you can move the folder.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Set

from datasets import load_dataset
from PIL import Image  # noqa: F401 (import ensures Pillow is available)
from tqdm import tqdm


datasets={"textvqa":"facebook/textvqa",
          "hme100k":"sionic-ai/hme100k",
          "vqa":"vqa/vqa2",
          "okvqa":"vqa/okvqa",
          "gqa":"vqa/gqa",
          "docvqa":"eliolio/docvqa", #
          "chartqa":"HuggingFaceM4/ChartQA",}

def build_argparser():
    p = argparse.ArgumentParser()
    #add dataset name:
    p.add_argument("--dataset", type=str, default="textvqa", help="HuggingFace dataset name")
    p.add_argument("--out", type=Path, required=True, help="Output directory")
    p.add_argument("--splits", type=str, default="train,validation,test",
                   help="Comma-separated splits to export (choices: train,validation,test)")
    p.add_argument("--format", type=str, default="plain", choices=["plain", "llava"],
                   help="JSONL format: 'plain' or 'llava' conversation-style")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    p.add_argument("--max-samples", type=int, default=None,
                   help="For quick tests: limit total samples per split")
    return p


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_image_once(sample: Dict[str, Any], images_dir: Path, saved_ids: Set[str]) -> str:
    """
    Saves the PIL image for this sample's image_id if not already saved.
    Returns the relative path 'images/<image_id>.jpg'
    """
    image_id = sample.get("image_id")
    if not image_id:
        # Fallback: derive a filename using question_id to ensure uniqueness
        image_id = f"qid_{sample.get('question_id')}"
    rel_path = Path("images") / f"{image_id}.jpg"
    abs_path = images_dir / f"{image_id}.jpg"
    if image_id not in saved_ids:
        img = sample["image"]  # PIL image from HF dataset
        img.save(abs_path, format="JPEG")
        saved_ids.add(image_id)
    return str(rel_path.as_posix())


def to_plain_record(sample: Dict[str, Any], rel_image_path: str, split: str) -> Dict[str, Any]:
    return {
        "id": sample.get("question_id"),
        "split": split,
        "image": rel_image_path,
        "question": sample["question"],
        # keep all 10 answers (may be empty strings for test set)
        "answers": sample.get("answers", []),
        "image_id": sample.get("image_id"),
        "image_width": sample.get("image_width"),
        "image_height": sample.get("image_height"),
        "image_classes": sample.get("image_classes"),
    }


def to_llava_record(sample: Dict[str, Any], rel_image_path: str, split: str) -> Dict[str, Any]:
    # Use the first non-empty answer as assistant reply (if any)
    answers = [a for a in sample.get("answers", []) if isinstance(a, str) and a.strip()]
    assistant_text = answers[0] if answers else ""
    return {
        "id": f"{split}_{sample.get('question_id')}",
        "image": rel_image_path,
        "conversations": [
            {"from": "human", "value": "<image>\n" + sample["question"]},
            {"from": "gpt", "value": assistant_text},
        ],
        "meta": {
            "split": split,
            "image_id": sample.get("image_id"),
        },
    }


def main():
    args = build_argparser().parse_args()
    out_dir: Path = args.out
    ensure_dir(out_dir)
    images_dir = out_dir / "images"
    ensure_dir(images_dir)

    jsonl_path = out_dir / ("annotations.jsonl" if args.format == "plain" else "conversations.jsonl")
    if jsonl_path.exists() and not args.overwrite:
        raise SystemExit(f"{jsonl_path} exists. Use --overwrite to replace.")
    
    # Load once; this will download the data on first run
    data = datasets[args.dataset]
    print(f"Loading dataset '{data}' from HuggingFace...")
    #if args.dataset == "docvqa":
    #    ds_dict = load_dataset('lmms-lab/DocVQA', 'DocVQA') #InfographicVQA??
    #else:
    #    ds_dict = load_dataset(data)
    ds_dict = load_dataset(data)
    # Keep track of which image_ids we've saved
    saved_ids: Set[str] = set()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    with jsonl_path.open("w", encoding="utf-8") as f_out:
        for split in splits:
            if split not in ds_dict:
                print(f"[WARN] Split '{split}' not found; skipping.")
                continue
            ds = ds_dict[split]
            n = len(ds) if args.max_samples is None else min(args.max_samples, len(ds))
            print(f"Processing split '{split}' with {n} samples...")
            for sample in tqdm(ds.select(range(n))):
                rel_img = save_image_once(sample, images_dir, saved_ids)
                if args.format == "plain":
                    rec = to_plain_record(sample, rel_img, split)
                else:
                    rec = to_llava_record(sample, rel_img, split)
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Done.\nImages: {images_dir}\nJSONL: {jsonl_path}")


if __name__ == "__main__":
    main()
