import os
import json
import argparse
from PIL import Image
import tqdm

# Default dataset directories
DS_DIRS = ["raw_data/llava_multi"]#"raw_data/chartqa_llava", "raw_data/docvqa_llava", "raw_data/textvqa_llava"]

# Resolutions to generate
TARGET_SIZES = [384, 768, 1024]


def save_resized(img, path_out, size):
    """Save resized version of image keeping aspect ratio, max side = size."""
    w, h = img.size
    if w > size or h > size:
        img_copy = img.copy()
        img_copy.thumbnail((size, size), Image.LANCZOS)
    else:
        img_copy = img.copy()
    img_copy.save(path_out, quality=95)


def process_dataset(ds_dir, output_root):
    img_dir = os.path.join(ds_dir, "images")

    # output path: output_root/DS_DIR/images
    out_ds_dir = os.path.join(output_root, os.path.basename(ds_dir))
    out_img_dir = os.path.join(out_ds_dir, "images")
    os.makedirs(out_img_dir, exist_ok=True)

    # Process images
    for fname in tqdm.tqdm(os.listdir(img_dir), desc=f"Processing {ds_dir}"):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(img_dir, fname)
        name, _ = os.path.splitext(fname)

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Could not process {path}: {e}")
            continue

        # Save multiple resolutions
        for size in TARGET_SIZES:
            out_name = f"{name}_{size}.jpg"  # normalize to jpg
            out_path = os.path.join(out_img_dir, out_name)
            if os.path.exists(out_path):
                continue
            save_resized(img, out_path, size)

    # Update conversations.jsonl
    conv_path = os.path.join(ds_dir, "conversations.jsonl")
    if os.path.exists(conv_path):
        new_conv_path = os.path.join(out_ds_dir, "conversations.jsonl")
        with open(conv_path, "r") as f_in, open(new_conv_path, "w") as f_out:
            for line in f_in:
                entry = json.loads(line)
                #check if entry is a list
                if isinstance(entry, list):
                    #make list of dicts into one dict
                    entry = {k: v for d in entry for k, v in d.items()}
                img_name = os.path.splitext(os.path.basename(entry["image"]))[0]
                entry["image"] = [
                    os.path.abspath(os.path.join(out_img_dir, f"{img_name}_{size}.jpg"))
                    for size in TARGET_SIZES
                ]
                f_out.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="output_dir",
        help="Root directory to save processed datasets"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for ds in DS_DIRS:
        process_dataset(ds, args.output_dir)
