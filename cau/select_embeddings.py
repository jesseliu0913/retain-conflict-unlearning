"""Remove retain examples nearest the forget centroid in a supplied embedding space."""

import argparse
import json
from collections import defaultdict, deque
import numpy as np
from .data import read_records, write_records


def record_key(record):
    return json.dumps(
        {k: record.get(k, "") for k in ("instruction", "input", "output")}, sort_keys=True
    )


def select_by_embedding(train, forget, retain, embeddings, near_percentile=10):
    if embeddings.ndim != 2 or len(embeddings) != len(train) or not np.isfinite(embeddings).all():
        raise ValueError("Expected finite embeddings with one row per training record.")
    if not 0 < near_percentile < 100 or not forget or not retain:
        raise ValueError("Use nonempty splits and a near percentile strictly between 0 and 100.")
    indices = defaultdict(deque)
    for i, record in enumerate(train):
        indices[record_key(record)].append(i)

    def locate(rows):
        found = []
        for row in rows:
            bucket = indices[record_key(row)]
            if not bucket:
                raise ValueError("Split records could not be matched to disjoint training rows.")
            found.append(bucket.popleft())
        return found

    forget_ids, retain_ids = locate(forget), locate(retain)
    centroid = embeddings[forget_ids].mean(axis=0)
    distances = np.linalg.norm(embeddings[retain_ids] - centroid, axis=1)
    cutoff = np.percentile(distances, near_percentile)
    kept = [row for row, distance in zip(retain, distances) if distance > cutoff]
    if not kept:
        raise ValueError("The distance threshold removed every retain record.")
    return kept


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("train", "forget", "retain", "embeddings", "output"):
        p.add_argument("--" + flag, required=True)
    p.add_argument("--near-percentile", type=float, default=10)
    args = p.parse_args()
    selected = select_by_embedding(
        read_records(args.train),
        read_records(args.forget),
        read_records(args.retain),
        np.load(args.embeddings),
        args.near_percentile,
    )
    write_records(selected, args.output)
    print(f"Kept {len(selected)} retain examples.")


if __name__ == "__main__":
    main()
