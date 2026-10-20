"""Select compatible retain records or mask conflicting numeric features."""

import argparse
import math
import re
import statistics
from .data import read_records, write_records


NUM_RE = re.compile(r"([A-Za-z0-9_ ]+) is ([+-]?\d+(?:\.\d+)?)")


ALT_NUM_RE = re.compile(r"([A-Za-z0-9_ ]+)\s*:\s*([+-]?\d+(?:\.\d+)?)")


PAIR_RE = re.compile(r"([A-Za-z0-9_ ]+)\s+is\s+([^,]+)")


def parse_features(text):
    features = {}
    for match in NUM_RE.finditer(text):
        key, val = match.group(1), match.group(2)
        features[key.strip()] = float(val)
    if features:
        return features
    for match in ALT_NUM_RE.finditer(text):
        key, val = match.group(1), match.group(2)
        features[key.strip()] = float(val)
    return features


def extract_feature_rows(records):
    rows = []
    missing = 0
    for idx, rec in enumerate(records):
        text = rec.get("input", "")
        feats = parse_features(text)
        if not feats:
            missing += 1
            continue
        rows.append((idx, feats))
    return rows, missing


def collect_feature_keys(rows):
    keys = set()
    for _, feats in rows:
        keys.update(feats.keys())
    return sorted(keys)


def stats_for_rows(rows, keys):
    per_key = {k: [] for k in keys}
    for _, feats in rows:
        for k in keys:
            if k in feats:
                per_key[k].append(feats[k])
    stats = {}
    for k, vals in per_key.items():
        if not vals:
            continue
        mean = statistics.fmean(vals)
        std = statistics.pstdev(vals)
        stats[k] = {"mean": mean, "std": std, "count": len(vals)}
    return stats


def normalized_distance(feats, stats, keys, min_std):
    total = 0.0
    count = 0
    for k in keys:
        if k not in feats or k not in stats:
            continue
        mean = stats[k]["mean"]
        std = stats[k]["std"] or 0.0
        if std < min_std:
            std = min_std
        z = (feats[k] - mean) / std
        total += z * z
        count += 1
    if count == 0:
        return None
    return math.sqrt(total / count)


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    if pct <= 0:
        return values[0]
    if pct >= 100:
        return values[-1]
    k = (len(values) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    d0 = values[int(f)] * (c - k)
    d1 = values[int(c)] * (k - f)
    return d0 + d1


def effect_sizes(forget_stats, retain_stats):
    effects = {}
    for k, fstats in forget_stats.items():
        rstats = retain_stats.get(k)
        if not rstats:
            continue
        pooled = math.sqrt((fstats["std"] ** 2 + rstats["std"] ** 2) / 2.0)
        if pooled == 0:
            d = 0.0
        else:
            d = (rstats["mean"] - fstats["mean"]) / pooled
        effects[k] = d
    return effects


def parse_input_pairs(input_text):
    if input_text.startswith("Patient information: "):
        input_text = input_text[len("Patient information: ") :]
    pairs = []
    for match in PAIR_RE.finditer(input_text):
        key = match.group(1).strip()
        value = match.group(2).strip()
        pairs.append((key, value))
    return pairs


def reconstruct_input(pairs):
    parts = [f"{key} is {value}" for key, value in pairs]
    return "Patient information: " + ", ".join(parts).rstrip(".") + "."


def mask_columns_in_record(record, columns_to_mask):
    text = record.get("input", "")
    pairs = parse_input_pairs(text)
    if not pairs:
        return record
    # Strip whitespace from column names for matching
    columns_to_mask = set(col.strip() for col in columns_to_mask)
    new_pairs = []
    for key, value in pairs:
        if key in columns_to_mask:
            new_pairs.append((key, "unknown"))
        else:
            new_pairs.append((key, value))
    record["input"] = reconstruct_input(new_pairs)

    # Update retain_columns to exclude masked columns
    if "retain_columns" in record:
        record["retain_columns"] = [
            col for col in record["retain_columns"] if col.strip() not in columns_to_mask
        ]

    return record


def select_retain(forget, retain, mode="sample", distance_percentile=90.0, top_k=1, min_std=1e-6):
    if not 0 <= distance_percentile <= 100 or top_k < 0 or min_std <= 0:
        raise ValueError("Invalid selection parameters.")
    forget_rows, missing_forget = extract_feature_rows(forget)
    retain_rows, missing_retain = extract_feature_rows(retain)
    if missing_forget or missing_retain:
        raise ValueError("Every record must contain numeric features in its input field.")
    keys = collect_feature_keys(forget_rows + retain_rows)
    fstats, rstats = (
        stats_for_rows(forget_rows, keys),
        stats_for_rows(retain_rows, keys),
    )
    if not fstats or not rstats:
        raise ValueError("Forget and retain sets must contain numeric data.")
    if mode == "feature":
        effects = effect_sizes(fstats, rstats)
        ranked = sorted(effects, key=lambda k: abs(effects[k]), reverse=True)
        columns = ranked[:top_k]
        return [mask_columns_in_record(dict(r), columns) for r in retain], {
            "masked_columns": columns
        }
    distances = [normalized_distance(row, fstats, keys, min_std) for _, row in forget_rows]
    cutoff = percentile(distances, distance_percentile)
    kept = []
    for index, row in retain_rows:
        distance = normalized_distance(row, fstats, keys, min_std)
        if distance is None:
            raise ValueError("A retain record has no features shared with the forget set.")
        if distance > cutoff:
            kept.append(retain[index])
    if not kept:
        raise ValueError("Selection removed every retain record; adjust the distance percentile.")
    return kept, {
        "distance_cutoff": cutoff,
        "kept": len(kept),
        "removed": len(retain) - len(kept),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--forget", required=True)
    p.add_argument("--retain", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=["sample", "feature"], default="sample")
    p.add_argument("--distance-percentile", type=float, default=90)
    p.add_argument("--top-k", type=int, default=1)
    args = p.parse_args()
    selected, info = select_retain(
        read_records(args.forget),
        read_records(args.retain),
        args.mode,
        args.distance_percentile,
        args.top_k,
    )
    write_records(selected, args.output)
    print(info)


if __name__ == "__main__":
    main()
