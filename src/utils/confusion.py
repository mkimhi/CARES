import json
import argparse
from collections import defaultdict
import math

RES_KEYS = ("384", "768", "1024")
RES_VALS = {"384": 384, "768": 768, "1024": 1024}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def find_res_dict(sample):
    """
    Find sub-dictionary containing any of RES_KEYS.
    Returns that dict or None.
    """
    # Top-level has some resolution keys
    if any(k in sample for k in RES_KEYS):
        return sample

    # Search nested dicts
    for v in sample.values():
        if isinstance(v, dict) and any(k in v for k in RES_KEYS):
            return v

    return None


def normalize_sample(sample):
    """
    Return a dict with ALL resolutions present.
    - If all 3 keys exist: use them.
    - If exactly 2 exist: linearly interpolate the missing one.
    - If <2 exist: raise ValueError (cannot fix).
    """
    res_dict = find_res_dict(sample)
    if res_dict is None:
        raise ValueError("No resolution keys in sample")

    present = [k for k in RES_KEYS if k in res_dict]

    # All present
    if len(present) == 3:
        return {k: float(res_dict[k]) for k in RES_KEYS}

    # Too few to interpolate
    if len(present) < 2:
        raise ValueError(f"Too few resolutions present: {present}")

    # Exactly 2 present -> interpolate the third
    missing = [k for k in RES_KEYS if k not in present][0]
    k1, k2 = present

    x1, y1 = RES_VALS[k1], float(res_dict[k1])
    x2, y2 = RES_VALS[k2], float(res_dict[k2])
    xm = RES_VALS[missing]

    if x2 == x1:
        ym = (y1 + y2) / 2.0
    else:
        ym = y1 + (y2 - y1) * (xm - x1) / (x2 - x1)

    full = {k1: y1, k2: y2, missing: ym}
    return {k: full[k] for k in RES_KEYS}


def build_normalized_map(data):
    """
    For a JSON dict {image_id: sample}, return:
      - normalized: {image_id: {res: anls}}
      - skipped: number of samples that couldn't be normalized
    """
    normalized = {}
    skipped = 0
    for img_id, sample in data.items():
        try:
            normalized[img_id] = normalize_sample(sample)
        except ValueError:
            skipped += 1
    return normalized, skipped


def compute_sufficient_resolution(scores, threshold, delta):
    """
    scores: dict with keys "384","768","1024" -> ANLS
    threshold: minimal ANLS to be considered "sufficient"
    delta: required improvement (>delta means next resolution is meaningfully better)

    Rule:
      - iterate resolutions in ascending order
      - a resolution r is sufficient if:
          scores[r] >= threshold
          AND (next_res_score - scores[r]) <= delta
      - for the largest resolution (1024), only threshold is checked.
      - if none satisfy, return the resolution with max ANLS (always one of the three).
    """
    order = ["384", "768", "1024"]

    for i, r in enumerate(order):
        s_r = scores[r]
        if s_r < threshold:
            continue

        if i < len(order) - 1:
            r_next = order[i + 1]
            s_next = scores[r_next]
            if s_next - s_r <= delta:
                return r
        else:
            # highest resolution: no "next"
            return r

    # Fallback: take the resolution with max ANLS
    return max(order, key=lambda k: scores[k])


def build_sufficient_map(norm_map, threshold, delta):
    """
    norm_map: {image_id: {res: anls}}
    Returns {image_id: sufficient_res} where sufficient_res in RES_KEYS.
    """
    res_map = {}
    for img_id, scores in norm_map.items():
        res_map[img_id] = compute_sufficient_resolution(scores, threshold, delta)
    return res_map


def build_confusion_matrix(map1, map2):
    """
    map1, map2: {image_id: res_label}
    Returns:
      - matrix: confusion matrix counts as dict[(r1, r2)] -> count
      - n_common: number of images present in BOTH maps
      - agreeing_ids: list of image_ids where map1 and map2 agree
    """
    common_ids = set(map1.keys()) & set(map2.keys())
    matrix = defaultdict(int)
    agreeing_ids = []

    for img_id in common_ids:
        r1 = map1[img_id]
        r2 = map2[img_id]
        matrix[(r1, r2)] += 1
        if r1 == r2:
            agreeing_ids.append(img_id)

    return matrix, len(common_ids), agreeing_ids


def print_confusion_matrix(matrix, n_common, n_full):
    """
    Print confusion matrix with:
      - Raw counts
      - Fractions relative to ALL images in the subset (n_full)
    """
    print("\n=== Confusion matrix of sufficient resolutions (on subset) ===")
    print("(rows=file1, cols=file2)")
    print(f"Images compared (subset)  : {n_common}")
    print(f"Total images in subset    : {n_full}\n")

    # Header
    header = ["file1 \\ file2"] + list(RES_KEYS)
    print("{:>15} ".format(header[0]), end="")
    for c in RES_KEYS:
        print("{:>16}".format(c), end="")
    print("\n")

    # Rows
    for r1 in RES_KEYS:
        print("{:>15} ".format(r1), end="")
        for r2 in RES_KEYS:
            count = matrix.get((r1, r2), 0)
            frac = count / n_full if n_full > 0 else 0
            print("{:>7d} ({:>7.3f})".format(count, frac), end="")
        print()


def compute_label_stats(matrix, n_common):
    """
    Compute:
      - percentage agreement
      - Pearson correlation (labels encoded as 0,1,2 for 384,768,1024)
      - Mutual information (bits) between the two labelings
    """
    if n_common == 0:
        return 0.0, 0.0, 0.0

    # Percent agreement
    total_agree = sum(matrix.get((r, r), 0) for r in RES_KEYS)
    percent_agree = total_agree / n_common

    # Build joint distribution p(x,y) from confusion matrix
    num_classes = len(RES_KEYS)
    res_to_idx = {r: i for i, r in enumerate(RES_KEYS)}

    p_xy = [[0.0 for _ in range(num_classes)] for _ in range(num_classes)]
    for r1 in RES_KEYS:
        for r2 in RES_KEYS:
            i = res_to_idx[r1]
            j = res_to_idx[r2]
            count = matrix.get((r1, r2), 0)
            p_xy[i][j] = count / n_common

    # Marginals
    p_x = [sum(row) for row in p_xy]
    p_y = [sum(p_xy[i][j] for i in range(num_classes)) for j in range(num_classes)]

    # Moments for Pearson
    # Labels encoded as 0,1,2
    E_x = sum(i * p_x[i] for i in range(num_classes))
    E_y = sum(j * p_y[j] for j in range(num_classes))
    E_x2 = sum((i ** 2) * p_x[i] for i in range(num_classes))
    E_y2 = sum((j ** 2) * p_y[j] for j in range(num_classes))
    E_xy = sum(i * j * p_xy[i][j] for i in range(num_classes) for j in range(num_classes))

    var_x = E_x2 - E_x ** 2
    var_y = E_y2 - E_y ** 2

    if var_x <= 0 or var_y <= 0:
        pearson = 0.0
    else:
        pearson = (E_xy - E_x * E_y) / math.sqrt(var_x * var_y)

    # Mutual information (in bits)
    mi = 0.0
    for i in range(num_classes):
        for j in range(num_classes):
            if p_xy[i][j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i][j] * math.log(p_xy[i][j] / (p_x[i] * p_y[j]), 2)

    return percent_agree, pearson, mi


def main():
    parser = argparse.ArgumentParser(
        description="Compare sufficient resolutions between two ANLS JSON files."
    )
    parser.add_argument("--file1", type=str, help="First JSON file")
    parser.add_argument("--file2", type=str, help="Second JSON file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,#0.4,
        help="ANLS threshold to consider a resolution sufficient (default: 0.5)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.1,#25,
        help="Max allowed improvement at next resolution (default: 0.1)",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=500,
        help="Max subset size for confusion/stats (0 means use all common images, default: 500)",
    )

    args = parser.parse_args()

    data1 = load_json(args.file1)
    data2 = load_json(args.file2)

    norm1, skipped1 = build_normalized_map(data1)
    norm2, skipped2 = build_normalized_map(data2)

    print(f"File1: {len(norm1)} usable samples, {skipped1} skipped (could not normalize)")
    print(f"File2: {len(norm2)} usable samples, {skipped2} skipped (could not normalize)")

    suff1 = build_sufficient_map(norm1, args.threshold, args.delta)
    suff2 = build_sufficient_map(norm2, args.threshold, args.delta)

    # Global agreement info
    _, n_common_all, agreeing_ids_all = build_confusion_matrix(suff1, suff2)
    common_ids = set(suff1.keys()) & set(suff2.keys())
    disagreeing_ids_all = sorted(list(common_ids - set(agreeing_ids_all)))
    agreeing_ids_all = sorted(agreeing_ids_all)

    print(f"\nTotal common images (both files) : {n_common_all}")
    print(f"Total agreeing images            : {len(agreeing_ids_all)}")
    print(f"Total disagreeing images         : {len(disagreeing_ids_all)}")

    if n_common_all == 0:
        print("\nNo common images to compare.")
        return

    # Determine effective subset size
    if args.subset_size is None or args.subset_size <= 0:
        effective_subset_size = n_common_all
    else:
        effective_subset_size = min(args.subset_size, n_common_all)

    # Fill subset: first all agreeing (up to capacity), then disagreements
    k_agree = min(len(agreeing_ids_all), effective_subset_size)
    k_disagree = effective_subset_size - k_agree

    subset_ids = agreeing_ids_all[:k_agree] + disagreeing_ids_all[:k_disagree]

    print(f"\nUsing subset size                : {effective_subset_size}")
    print(f"  In subset: agreeing            : {k_agree}")
    print(f"           : disagreeing         : {k_disagree}")

    # Restrict maps to subset
    subset_map1 = {img_id: suff1[img_id] for img_id in subset_ids}
    subset_map2 = {img_id: suff2[img_id] for img_id in subset_ids}

    matrix_subset, n_subset, agreeing_in_subset = build_confusion_matrix(subset_map1, subset_map2)
    n_full_subset = n_subset  # by construction, n_subset == len(subset_ids)

    print_confusion_matrix(matrix_subset, n_subset, n_full_subset)

    percent_agree, pearson, mi = compute_label_stats(matrix_subset, n_subset)
    print("\n=== Agreement statistics (on ordered subset) ===")
    print(f"Subset size             : {n_subset}")
    print(f"Total agreement         : {percent_agree * 100:.2f}%")
    print(f"Pearson correlation     : {pearson:.4f}")
    print(f"Mutual information (bit): {mi:.4f}")
    print(f"Agreeing in subset      : {len(agreeing_in_subset)} / {n_subset}")

    print("\nExample IDs from subset:", subset_ids[:10])


if __name__ == "__main__":
    main()
