import os, io
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import jsonlines
import tqdm

# List of dataset directories
DS_DIRS = ["chartqa_llava", "docvqa_llava", "textvqa_llava"]

def get_img(path):
    """Open image safely"""
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"Could not open {path}: {e}")
        return None

for DS_DIR in DS_DIRS:
    name = os.path.basename(DS_DIR.rstrip("/"))
    img_dir = os.path.join(DS_DIR, "images")

    widths, heights, examples = [], [], []

    # Collect resolution info
    for fname in tqdm.tqdm(os.listdir(img_dir), desc=f"Processing {name}"):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(img_dir, fname)
            img = get_img(path)
            if img is None:
                continue
            w, h = img.size
            is_highres = (w > 500) and (h > 500)
            widths.append(w)
            heights.append(h)
            examples.append({"file": fname, "w": w, "h": h, "is_highres": is_highres})

    # Count resolution occurrences
    res_counts = Counter(zip(widths, heights))
    x = [w for (w, h) in res_counts.keys()]
    y = [h for (w, h) in res_counts.keys()]
    sizes = [count for count in res_counts.values()]
    size_scale = 20
    sizes_scaled = [s * size_scale for s in sizes]

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=sizes_scaled, alpha=0.2, edgecolors='w')
    plt.xlabel("Width (px)")
    plt.ylabel("Height (px)")
    plt.title(f"Image Resolution Distribution: {name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{name}_resolution_histogram.png", dpi=300)
    plt.close()
    print(f"Saved histogram for {name} -> {name}_resolution_histogram.png")

    # Print summary
    total = len(examples)
    high = sum(e["is_highres"] for e in examples)
    low = total - high
    print(f"[{name}] Total: {total}, High-res: {high/total:.2%}, Low/medium: {low/total:.2%}")

    # Save dataset stats to jsonl
    out_jsonl = f"{name}.jsonl"
    with jsonlines.open(out_jsonl, mode='w') as writer:
        for e in examples:
            writer.write(e)
    print(f"Saved dataset metadata for {name} -> {out_jsonl}")


# import io, os, math, csv
# from datasets import load_dataset, concatenate_datasets, DatasetDict
# from PIL import Image
# import matplotlib.pyplot as plt
# from collections import Counter, defaultdict

# # Helper to check resolution
# def img_size(example, img_col="image"):
#     img = example[img_col]
#     if isinstance(img, dict) and "bytes" in img:
#         img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
#     w, h = img.size
#     example["w"], example["h"] = w, h
#     example["is_highres"] = (w > 1000) and (h > 1000)
#     return example

# def get_img(imgobj):
#     # HF datasets returns a PIL.Image.Image directly in most cases,
#     # but handle byte dict just in case.
#     if isinstance(imgobj, Image.Image):
#         return imgobj
#     if isinstance(imgobj, dict) and "bytes" in imgobj:
#         return Image.open(io.BytesIO(imgobj["bytes"])).convert("RGB")
#     return imgobj  # best effort


# name = 'InfographicVQA_20k+TextCaps_10k'

# # InfographicVQA
# infovqa = load_dataset("LIME-DATA/infovqa")["train"]
# infovqa = infovqa.map(img_size, num_proc=4)
# infovqa_20k = infovqa.select(range(min(20000, len(infovqa))))

# # TextCaps
# textcaps = load_dataset("lmms-lab/TextCaps", "default")["train"]
# textcaps = textcaps.map(img_size, num_proc=4)
# textcaps_10k = textcaps.select(range(min(10000, len(textcaps))))

# # Merge
# mix_train = concatenate_datasets([
#     infovqa_20k.shuffle(seed=42),
#     textcaps_10k.shuffle(seed=42)
# ])
# mix = DatasetDict({"train": mix_train})



# print('Collect statistics for', name)

# # Count resolution occurrences
# widths = mix["train"]["w"]
# heights = mix["train"]["h"]
# res_counts = Counter(zip(widths, heights))
# # Prepare plotting data
# x = [w for (w, h) in res_counts.keys()]
# y = [h for (w, h) in res_counts.keys()]
# sizes = [count for count in res_counts.values()]
# size_scale = 20
# sizes_scaled = [s * size_scale for s in sizes]

# # Plot & save
# plt.figure(figsize=(8, 6))
# plt.scatter(x, y, s=sizes_scaled, alpha=0.2, edgecolors='w')
# plt.xlabel("Width (px)")
# plt.ylabel("Height (px)")
# plt.title(f"Image Resolution Distribution {name}")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(f"{name}_resolution_histogram.png", dpi=300)
# print("Saved figure as resolution_histogram.png")

# # # Inspect high/low counts (optional)
# total = len(mix["train"])
# high = sum(mix["train"]["is_highres"])
# low  = total - high
# print("Total:", len(mix["train"]), "High-res:", high/total, "Low/medium:", low/total)


# ##save into jsonl:
# import jsonlines
# print(f"Saving dataset to {name}.jsonl....")
# with jsonlines.open(f"{name}.jsonl", mode='w') as writer:
#     for example in mix["train"]:
#         print('------')
#         print(example)
#         print('------')
#         writer.write(example)
# print(f"Saved dataset to {name}.jsonl")


# plt.figure(figsize=(8, 6))
# plt.scatter(x, y, s=sizes_scaled, alpha=0.2, edgecolors='w')
# plt.xlabel("Width (px)")
# plt.ylabel("Height (px)")
# plt.title(f"Image Resolution Distribution {name}")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(f"{name}_resolution_histogram.png", dpi=300)
# print("Saved figure as resolution_histogram.png")

# # # Inspect high/low counts (optional)
# total = len(mix["train"])
# high = sum(mix["train"]["is_highres"])
# low  = total - high
# print("Total:", len(mix["train"]), "High-res:", high/total, "Low/medium:", low/total)


# ##save into jsonl:
# import jsonlines
# print(f"Saving dataset to {name}.jsonl....")
# with jsonlines.open(f"{name}.jsonl", mode='w') as writer:
#     for example in mix["train"]:
#         print('------')
#         print(example)
#         print('------')
#         writer.write(example)
# print(f"Saved dataset to {name}.jsonl")
