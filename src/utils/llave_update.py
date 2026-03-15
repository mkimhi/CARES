
import json
import re
import argparse
from typing import Dict, Any, List, Tuple
from collections import Counter

def pick_resolution(scores: Dict[str, float], hi_thr: float = 0.8, margin: float = 0.1) -> int:
    s384 = float(scores.get("384", 0.0))
    s768 = float(scores.get("768", 0.0))
    s1024 = float(scores.get("1024", 0.0))

    if s384 > hi_thr:
        return 384
    elif s768>hi_thr:
        return 768
    else:
        return 1024
    # elif (s768 - s384) >= margin and (s1024 - s768) <= margin:
    #     return 768
    # elif (s1024 - s384) > margin or (s1024 - s384) > margin:
    #     # second clause duplicates the first per the user's spec
    #     return 1024
    # else:
    #     return 384

_RES_RX = re.compile(r"^(?P<prefix>.+?)_(?P<res>384|768|1024)(?:_\d+)?\.jpg$", re.IGNORECASE)

def build_image_triplet(images: List[str]) -> List[str]:
    """
    Try to infer a base path from any existing *_384.jpg|*_768.jpg|*_1024.jpg (optionally *_<res>_<n>.jpg).
    Then return [base_384.jpg, base_768.jpg, base_1024.jpg].
    If no match, return the original list unchanged.
    """
    base = None
    for p in images:
        m = _RES_RX.match(p)
        if m:
            base = m.group("prefix")
            break
    if base is None:
        # fallback: try to coerce the first path by appending resolutions if it ends with .jpg
        if images and images[0].lower().endswith(".jpg"):
            stem = images[0]
            # Try to strip a trailing resolution-like segment even if regex failed (edge cases)
            m = _RES_RX.match(stem)
            if m:
                base = m.group("prefix")
            else:
                # remove extension and any trailing underscores digits
                stem_no_ext = re.sub(r"\.jpg$", "", stem, flags=re.IGNORECASE)
                stem_no_ext = re.sub(r"(_\d+)$", "", stem_no_ext)
                base = stem_no_ext  # may duplicate dirs, but best effort
        else:
            return images  # give up
    triplet = [f"{base}_{r}.jpg" for r in (384, 768, 1024)]
    return triplet

def update_item(item: Dict[str, Any], scores: Dict[str, Dict[str, float]], hi_thr: float, margin: float) -> Dict[str, Any]:
    item_id = item.get("id")
    sc = scores.get(item_id, {})
    chosen = pick_resolution(sc, hi_thr=hi_thr, margin=margin)

    # 1) change first GPT answer under conversations
    convs = item.get("conversations", [])
    for turn in convs:
        if isinstance(turn, dict) and turn.get("from") == "gpt":
            turn["value"] = f"<RESOLUTION value={chosen} />"
            break
    item["conversations"] = convs

    # 2) change sufficient_res
    item["sufficient_res"] = chosen

    # 3) normalize image list to the 3 images
    images = item.get("image", [])
    item["image"] = build_image_triplet(images)

    return item

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to input data.json (LLaVA format list)")
    ap.add_argument("--scores", required=True, help="Path to ANLS scores JSON (id -> {384,768,1024})")
    ap.add_argument("--out", required=True, help="Path to save updated JSON")
    ap.add_argument("--hi", type=float, default=0.8, help="High score threshold for 384 (default: 0.8)")
    ap.add_argument("--margin", type=float, default=0.1, help="Margin threshold (default: 0.1)")
    args = ap.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(args.scores, "r", encoding="utf-8") as f:
        scores = json.load(f)

    if not isinstance(data, list):
        raise ValueError("data.json must be a list of items")

    out_data = []
    for item in data:
        out_data.append(update_item(item, scores, args.hi, args.margin))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    with open(args.out, "r", encoding="utf-8") as f:
        data = json.load(f)
 
    # collect all sufficient_res values
    resolutions = [item.get("sufficient_res") for item in data]

    # count occurrences
    counter = Counter(resolutions)

    print("Statistics of sufficient_res:")
    total = len(resolutions)
    for res, count in sorted(counter.items()):
        pct = (count / total) * 100
        print(f"Resolution {res}: {count} ({pct:.2f}%)")

    print(f"Total items: {total}")



if __name__ == "__main__":
    main()

