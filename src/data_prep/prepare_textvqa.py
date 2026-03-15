#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
from typing import Dict, Any, List, Set

from PIL import Image  # noqa: F401
from tqdm import tqdm
from datasets import load_dataset,concatenate_datasets, Image as HFImage
import random
random.seed(42)


DATA_ROOT = "/proj/mmfm/data/llava_665k_multi"
DATASETS = {
    "textvqa": "facebook/textvqa",
    "hme100k": "sionic-ai/hme100k",
    "vqa": "vqa/vqa2",
    "okvqa": "vqa/okvqa",
    "gqa": "vqa/gqa",
    "docvqa": "eliolio/docvqa",
    "chartqa": "chartqa",
    "llava_multi": ['TIGER-Lab/Mantis-Instruct','llava_665k_multi']
}

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="textvqa",
                   help="One of: " + ",".join(DATASETS.keys()))
    p.add_argument("--subset", type=str, default=None,
                   help="HF subset/config (for DocVQA use DocVQA or InfographicVQA).")
    p.add_argument("--out", type=Path, required=True, help="Output directory")
    p.add_argument("--splits", type=str, default="train,validation,test",
                   help="Comma-separated splits to export")
    p.add_argument("--format", type=str, default="plain", choices=["plain", "llava"],
                   help="Output JSONL format")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-samples", type=int, default=None)
    return p


def add_resolution(example):
    rel_path = example["images"][0]["path"]
    abs_path = os.path.join(DATA_ROOT, rel_path)
    with Image.open(abs_path) as img:
        w, h = img.size
    example["images"][0]["path"] = abs_path   # overwrite with absolute path
    example["width"] = w
    example["height"] = h
    example["max_dim"] = max(w, h)
    return example


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _docvqa_image_id(sample: Dict[str, Any]) -> str:
    """
    Prefer a stable per-page id. DocVQA provides docId and ucsf_document_page_no.
    Fallbacks to questionId if needed.
    """
    doc_id = sample.get("docId")
    page = sample.get("ucsf_document_page_no")
    qid = sample.get("questionId") or sample.get("question_id")
    if doc_id is not None and page is not None:
        return f"doc{doc_id}_p{page}"
    if qid is not None:
        return f"qid_{qid}"
    return os.urandom(4).hex()  # last resort

def get_image_id(sample: Dict[str, Any]) -> str:
    # Generic path; special-case DocVQA-style fields
    if any(k in sample for k in ("docId", "ucsf_document_page_no", "questionId")):
        return _docvqa_image_id(sample)
    # Otherwise try common fields
    return (
        sample.get("image_id")
        or f"qid_{sample.get('question_id')}"
        or os.urandom(4).hex()
    )

def save_image_once(sample: Dict[str, Any], images_dir: Path, saved_ids: Set[str],paths=False) -> str:
    """
    Saves the PIL image for this sample's image id if not already saved.
    Returns the relative path 'images/<image_id>.jpg'
    paths: if True, sample["images"] is a list of paths (llava_multi); use the first one (key is 'path').
    """
    image_id = get_image_id(sample)
    if paths:
        image_id = os.path.splitext(os.path.basename(sample["images"][0]['path']))[0].split('_')[-1]
    rel_path = Path("images") / f"{image_id}.jpg"
    abs_path = images_dir / rel_path.name
    if image_id not in saved_ids:
        if paths:
            img_path = sample["images"][0]["path"]
            img = Image.open(img_path)
        else:
            img = sample["image"]  # HF Image feature -> PIL.Image
        # ensure JPEG-compatible
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        elif img.mode == "L":
            img = img.convert("RGB")
        if paths:
            img.save(abs_path, format="JPEG", quality=95)
        else:
            img.save(abs_path, format="JPEG", quality=95)
        saved_ids.add(image_id)
    return rel_path.as_posix()

def to_plain_record(sample: Dict[str, Any], rel_image_path: str, split: str) -> Dict[str, Any]:
    # Prefer DocVQA field names; fallback to your originals
    qid = sample.get("questionId", sample.get("question_id"))
    return {
        "id": qid,
        "split": split,
        "image": rel_image_path,
        "question": sample.get("question", ""),
        "answers": sample.get("answers", []),
        "image_id": get_image_id(sample),
        # extra docvqa metadata if present
        "docId": sample.get("docId"),
        "ucsf_document_id": sample.get("ucsf_document_id"),
        "ucsf_document_page_no": sample.get("ucsf_document_page_no"),
        "question_types": sample.get("question_types"),
    }

def to_llava_record(sample: Dict[str, Any], rel_image_path: str, split: str,multi=False,id=None) -> Dict[str, Any]:
    qid = sample.get("questionId", sample.get("question_id",sample.get('id')))
    records = []
    if multi:
        qid = id if id is not None else qid
        qs=[]
        ans=[]
        for i in sample['conversation']:
            if isinstance(i.get('content',[]),str):
                if i.get('role',[])=='user':
                    qs.append(i['content'].replace('<image>',""))
                if i.get('role',[])=='gpt' or i.get('role',[])=='assistant':
                    ans.append(i['content'])
        for idx, (q, a) in enumerate(zip(qs, ans)):
            record = {
                "id": f"{split}_{qid}_{idx}",
                "image": rel_image_path,
                "conversations": [
                    {"from": "human", "value": "<image>\n" + q},
                    {"from": "gpt", "value": a},
                ],
                "meta": {
                    "split": split,
                    "image_id": f'{qid}_{idx}',
                    "docId": sample.get("docId"),
                    "ucsf_document_page_no": sample.get("ucsf_document_page_no"),
                },
            }
            records.append(record)
        return records
    else:
        answers = [a for a in sample.get("answers", []) if isinstance(a, str) and a.strip()]
        assistant_text = answers[0] if answers else ""
        return {
            "id": f"{split}_{qid}",
            "image": rel_image_path,
            "conversations": [
                {"from": "human", "value": "<image>\n" + sample.get("question", "")},
                {"from": "gpt", "value": assistant_text},
            ],
            "meta": {
                "split": split,
                "image_id": get_image_id(sample),
                "docId": sample.get("docId"),
                "ucsf_document_page_no": sample.get("ucsf_document_page_no"),
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

    # Load dataset
    hf_name = DATASETS[args.dataset]
    print(f"Loading dataset '{hf_name}' from Hugging Face...")
    if args.dataset == "llava_multi":
        ds_dict = load_dataset(DATASETS[args.dataset][0], DATASETS[args.dataset][1])#["train"]
        default_splits = ["train"]
    elif args.dataset == "docvqa":
        # Read the train Parquet shards directly from the HF repo
        ds_dict = load_dataset(
            "parquet",
            data_files={"train": "hf://datasets/lmms-lab/DocVQA/DocVQA/train-*.parquet"},
        )
        # Ensure the 'image' column is decoded as PIL
        if not isinstance(ds_dict["train"].features.get("image"), HFImage):
            ds_dict["train"] = ds_dict["train"].cast_column("image", HFImage())
        default_splits = ["train"]
    else:
        ds_dict = load_dataset(hf_name)
        default_splits = ["train", "validation", "test"]
    saved_ids: Set[str] = set()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()] or default_splits

    with jsonl_path.open("w", encoding="utf-8") as f_out:
        for split in splits:
            if split not in ds_dict:
                print(f"[WARN] Split '{split}' not found; skipping.")
                continue
            ds = ds_dict[split]
            if args.dataset == "llava_multi" and split == "train":
                # use only samples with single image
                print('filter llava_multi to single image samples')
                ds = ds.filter(lambda x: len(x['images']) == 1)
                ds = ds.map(add_resolution)
                dataset_sorted = ds.sort("max_dim", reverse=True)
                n = len(dataset_sorted) if args.max_samples is None else min(args.max_samples, len(dataset_sorted))
                ds_high = dataset_sorted.select(range(int(0.9*n)))
                rest = dataset_sorted.select(range(int(0.9*n), len(dataset_sorted)))
                indices = random.sample(range(len(rest)), 1000)
                ds_rest = rest.select(indices)
                ds = concatenate_datasets([ds_high, ds_rest])
                print(f"After filtering, {len(ds)} samples remain in 'llava_multi') train split.")
            else:
                n = len(ds) if args.max_samples is None else min(args.max_samples, len(ds))
                print(f"Processing split '{split}' with {n} samples...")
            # iterate deterministically over first n samples
            paths = args.dataset == "llava_multi"
            for ids, sample in tqdm(enumerate(ds.select(range(n)))):
                rel_img = save_image_once(sample, images_dir, saved_ids,paths=paths)
                rec = to_plain_record(sample, rel_img, split) if args.format == "plain" \
                      else to_llava_record(sample, rel_img, split,multi=paths,id=ids)
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Done.\nImages: {images_dir}\nJSONL: {jsonl_path}")

if __name__ == "__main__":
    main()
