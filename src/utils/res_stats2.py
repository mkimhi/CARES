import json
from collections import Counter
from itertools import combinations
from statistics import mean, pstdev

JSON_PATH = "/proj/mmfm/kimhi/data/data/mix2/resolutions.json"
RES_KEYS = ("384", "768", "1024")
RES_VALS = { "384": 384, "768": 768, "1024": 1024 }

def load_data(path):
    with open(path, "r") as f:
        return json.load(f)

def find_res_dict(sample):
    """
    Find the sub-dictionary that contains at least one of the resolution keys.
    """
    # Case 1: sample itself has any resolution keys
    if any(k in sample for k in RES_KEYS):
        return sample

    # Case 2: look in nested dicts
    for v in sample.values():
        if isinstance(v, dict) and any(k in v for k in RES_KEYS):
            return v

    # Nothing found
    return None

def normalize_sample(sample):
    """
    Return a dict with ALL resolution keys present.
    If exactly one is missing, interpolate it linearly from the other two.
    If fewer than 2 resolutions are present, raise ValueError.
    """
    res_dict = find_res_dict(sample)
    if res_dict is None:
        raise ValueError("No resolution keys found")

    # Collect the resolutions that actually appear
    present = [k for k in RES_KEYS if k in res_dict]
    if len(present) == 3:
        # Already complete – just project to the three keys
        return {k: float(res_dict[k]) for k in RES_KEYS}

    if len(present) < 2:
        # Can't interpolate with < 2 points
        raise ValueError(f"Too few resolutions present: {present}")

    # Exactly 2 present -> interpolate the missing one
    if len(present) == 2:
        missing = [k for k in RES_KEYS if k not in present][0]
        k1, k2 = present

        x1, y1 = RES_VALS[k1], float(res_dict[k1])
        x2, y2 = RES_VALS[k2], float(res_dict[k2])
        xm = RES_VALS[missing]

        # Linear interpolation: y = y1 + (y2 - y1) * (xm - x1) / (x2 - x1)
        if x2 == x1:
            # Should never happen with 384,768,1024 but guard anyway
            ym = (y1 + y2) / 2.0
        else:
            ym = y1 + (y2 - y1) * (xm - x1) / (x2 - x1)

        full = {
            k1: y1,
            k2: y2,
            missing: ym
        }
        # Ensure full order
        return {k: full[k] for k in RES_KEYS}

    # Fallback (shouldn’t be reached)
    raise ValueError("Unexpected number of present resolutions")

def build_samples(data):
    """
    Convert data.values() into a list of normalized samples with all 3 resolutions.
    Returns (samples, skipped_count).
    """
    samples = []
    skipped = 0
    for s in data.values():
        try:
            samples.append(normalize_sample(s))
        except ValueError:
            skipped += 1
    return samples, skipped

def compute_basic_stats(samples):
    n = len(samples)
    avg = {r: mean(s[r] for s in samples) for r in RES_KEYS}
    min_v = {r: min(s[r] for s in samples) for r in RES_KEYS}
    max_v = {r: max(s[r] for s in samples) for r in RES_KEYS}
    overall = mean(s[r] for s in samples for r in RES_KEYS)

    return {
        "n_samples": n,
        "avg_anls": avg,
        "min_anls": min_v,
        "max_anls": max_v,
        "overall_avg": overall,
    }

def compute_best_stats(samples):
    n = len(samples)
    best_counts = Counter()
    tie_counts = Counter()

    for s in samples:
        max_val = max(s[r] for r in RES_KEYS)
        best = [r for r in RES_KEYS if s[r] == max_val]
        if len(best) == 1:
            best_counts[best[0]] += 1
        else:
            tie_counts[frozenset(best)] += 1

    best_percent = {r: 100.0 * best_counts[r] / n for r in RES_KEYS}
    tie_percent = {"+".join(sorted(t)): 100.0 * c / n for t, c in tie_counts.items()}

    return {
        "best_counts": dict(best_counts),
        "best_percentages": best_percent,
        "tie_counts": {"+".join(sorted(t)): c for t, c in tie_counts.items()},
        "tie_percentages": tie_percent,
    }

def compute_pairwise_gaps(samples):
    stats = {}
    for r1, r2 in combinations(RES_KEYS, 2):
        diffs = [s[r2] - s[r1] for s in samples]
        abs_diffs = [abs(d) for d in diffs]
        stats[(r1, r2)] = {
            "avg_signed_gap_r2_minus_r1": mean(diffs),
            "avg_abs_gap": mean(abs_diffs),
            "std_signed_gap": pstdev(diffs) if len(diffs) > 1 else 0.0,
        }
    return stats

def compute_high_score_stats(samples, thresholds=(0.5, 0.7, 0.9)):
    n = len(samples)
    result = {"per_threshold": {}, "close_resolutions": {}}

    # percentage of samples above threshold per resolution
    for t in thresholds:
        result["per_threshold"][t] = {
            r: 100.0 * sum(s[r] >= t for s in samples) / n
            for r in RES_KEYS
        }

    # how often all resolutions are within a small margin
    for margin in (0.01, 0.03, 0.05, 0.1):
        close_count = 0
        for s in samples:
            scores = [s[r] for r in RES_KEYS]
            if max(scores) - min(scores) <= margin:
                close_count += 1
        result["close_resolutions"][margin] = 100.0 * close_count / n

    return result

def print_report(basic_stats, best_stats, pairwise_stats, high_score_stats, skipped):
    print("=== Dataset summary ===")
    print(f"Number of usable samples: {basic_stats['n_samples']}")
    print(f"Number of skipped samples (too few resolutions): {skipped}")
    print()

    print("=== Average ANLS per resolution ===")
    for r in RES_KEYS:
        print(f"  {r}: {basic_stats['avg_anls'][r]:.4f}")
    print(f"\nOverall average ANLS: {basic_stats['overall_avg']:.4f}")
    print()

    print("=== Min / Max ANLS per resolution ===")
    for r in RES_KEYS:
        print(f"  {r}: min={basic_stats['min_anls'][r]:.4f}, max={basic_stats['max_anls'][r]:.4f}")
    print()

    print("=== Best-resolution statistics (strict winners) ===")
    for r in RES_KEYS:
        count = best_stats["best_counts"].get(r, 0)
        perc = best_stats["best_percentages"].get(r, 0.0)
        print(f"  {r}: {count} samples ({perc:.2f}%)")
    print()

    if best_stats["tie_counts"]:
        print("=== Ties (multiple resolutions share best ANLS) ===")
        for combo, count in best_stats["tie_counts"].items():
            perc = best_stats["tie_percentages"][combo]
            print(f"  {combo}: {count} samples ({perc:.2f}%)")
        print()
    else:
        print("No ties.\n")

    print("=== Pairwise gaps between resolutions ===")
    for (r1, r2), st in pairwise_stats.items():
        print(f"  {r2} - {r1}:")
        print(f"    avg signed gap: {st['avg_signed_gap_r2_minus_r1']:.4f}")
        print(f"    avg absolute gap: {st['avg_abs_gap']:.4f}")
        print(f"    std of signed gap: {st['std_signed_gap']:.4f}")
    print()

    print("=== High-score stats (percentage of samples >= threshold) ===")
    for t, per_res in high_score_stats["per_threshold"].items():
        print(f"  Threshold {t}:")
        for r, perc in per_res.items():
            print(f"    {r}: {perc:.2f}%")
    print()

    print("=== How often resolutions are close to each other ===")
    for margin, perc in high_score_stats["close_resolutions"].items():
        print(f"  All resolutions within ±{margin:.2f}: {perc:.2f}% of samples")

def main():
    data = load_data(JSON_PATH)

    samples, skipped = build_samples(data)

    if not samples:
        raise RuntimeError("No valid samples after interpolation/normalization.")

    basic = compute_basic_stats(samples)
    best = compute_best_stats(samples)
    gaps = compute_pairwise_gaps(samples)
    high = compute_high_score_stats(samples)

    print_report(basic, best, gaps, high, skipped)

if __name__ == "__main__":
    main()
