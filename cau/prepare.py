"""Linearize tabular training data and construct a sample or feature forget request."""

import argparse
import csv
import re
from pathlib import Path
from .data import read_records, write_records
from .select import mask_columns_in_record


def feature_name(name):
    return re.sub(r"[^A-Za-z0-9_ ]", "_", name.strip())


def linearize(rows, target, instruction, exclude=()):
    records = []
    for row in rows:
        if target not in row:
            raise ValueError(f"Missing target column: {target}")
        columns = sorted(set(row) - {target} - set(exclude))
        text = ", ".join(f"{feature_name(key)} is {row[key]}" for key in columns)
        records.append(
            {
                "instruction": instruction,
                "input": f"Patient information: {text}.",
                "output": str(row[target]),
            }
        )
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Training rows only: CSV or JSON/JSONL records.")
    p.add_argument("--target", required=True)
    p.add_argument("--instruction", required=True)
    p.add_argument("--exclude", nargs="*", default=["id", "ID", "index"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mode", required=True, choices=["sample", "feature"])
    p.add_argument("--forget-columns", nargs="+")
    p.add_argument("--forget-clusters", nargs="+", type=int)
    p.add_argument("--clusters", type=int, default=10)
    p.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if Path(args.input).suffix == ".csv":
        with open(args.input) as f:
            rows = list(csv.DictReader(f))
    else:
        rows = read_records(args.input)
    records = linearize(rows, args.target, args.instruction, args.exclude)
    if args.mode == "feature":
        if not args.forget_columns or args.target in args.forget_columns:
            p.error("Specify feature names with --forget-columns, excluding the target.")
        missing = set(args.forget_columns) - set(rows[0])
        if missing:
            p.error(f"Unknown forget columns: {sorted(missing)}")
        forget = records
        retain = [
            mask_columns_in_record(dict(row), [feature_name(c) for c in args.forget_columns])
            for row in records
        ]
    else:
        if not args.forget_clusters or not all(
            0 <= c < args.clusters for c in args.forget_clusters
        ):
            p.error("Specify cluster IDs in [0, --clusters) with --forget-clusters.")
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        encoder = SentenceTransformer(args.encoder)
        vectors = encoder.encode(
            [r["instruction"] + " " + r["input"] for r in records], show_progress_bar=True
        )
        labels = KMeans(n_clusters=args.clusters, random_state=args.seed, n_init=10).fit_predict(
            StandardScaler().fit_transform(vectors)
        )
        forget, retain = [], []
        for record, label in zip(records, labels):
            record["cluster_id"] = int(label)
            (forget if label in args.forget_clusters else retain).append(record)
    if not forget or not retain:
        raise ValueError("Both forget and retain sets must be nonempty.")
    destination = Path(args.output_dir)
    write_records(forget, destination / "forget.json")
    write_records(retain, destination / "retain.json")
    print(f"Prepared {len(forget)} forget and {len(retain)} retain examples.")


if __name__ == "__main__":
    main()
