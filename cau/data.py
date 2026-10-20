"""Instruction data and paired forget/retain batches."""

import json
from pathlib import Path
import torch
from torch.utils.data import Dataset


def read_records(path):
    path = Path(path)
    with path.open() as f:
        rows = (
            [json.loads(line) for line in f if line.strip()]
            if path.suffix == ".jsonl"
            else json.load(f)
        )
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty JSON array or JSONL file.")
    return rows


def write_records(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        if path.suffix == ".jsonl":
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            json.dump(rows, f, indent=2, ensure_ascii=False)
            f.write("\n")


def prompt_for(record, tokenizer):
    question = f"{record.get('instruction', '')}\n\n{record.get('input', '')}".strip()
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"### Instruction:\n{question}\n\n### Response:\n"


def encode_record(record, tokenizer, max_length):
    prompt = tokenizer(prompt_for(record, tokenizer), add_special_tokens=False)["input_ids"]
    answer = tokenizer(str(record["output"]), add_special_tokens=False)["input_ids"]
    if not answer or max_length < 3:
        raise ValueError("A nonempty answer and max_length >= 3 are required.")
    answer = answer[: max_length - 2] + [tokenizer.eos_token_id]
    prompt = prompt[-(max_length - len(answer)) :]
    if not prompt:
        raise ValueError("A nonempty prompt is required.")
    ids = prompt + answer
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": [-100] * len(prompt) + answer,
    }


class PairedDataset(Dataset):
    def __init__(self, forget, retain, tokenizer, max_length):
        if not forget or not retain:
            raise ValueError("Forget and retain sets must both be nonempty.")
        self.forget = [encode_record(x, tokenizer, max_length) for x in forget]
        self.retain = [encode_record(x, tokenizer, max_length) for x in retain]

    def __len__(self):
        return len(self.forget)

    def __getitem__(self, index):
        retain_index = torch.randint(len(self.retain), ()).item()
        return {"forget": self.forget[index], "retain": self.retain[retain_index]}


class PairedCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        result = {}
        for split in ("forget", "retain"):
            length = max(len(row[split]["input_ids"]) for row in rows)
            result[split] = {}
            for key, pad in [
                ("input_ids", self.pad_token_id),
                ("attention_mask", 0),
                ("labels", -100),
            ]:
                result[split][key] = torch.tensor(
                    [row[split][key] + [pad] * (length - len(row[split][key])) for row in rows],
                    dtype=torch.long,
                )
        return result
