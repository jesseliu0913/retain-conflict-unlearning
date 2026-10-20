"""Measure exact-match accuracy and answer-token loss on a prepared split."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=64)
    args = p.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from .data import read_records, encode_record, prompt_for

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model)
        .to("cuda" if torch.cuda.is_available() else "cpu")
        .eval()
    )
    correct, losses, predictions = 0, [], []
    with torch.inference_mode():
        for row in read_records(args.data):
            encoded = {
                k: torch.tensor([v], device=model.device)
                for k, v in encode_record(row, tokenizer, args.max_length).items()
            }
            losses.append(float(model(**encoded).loss))
            prompt = tokenizer(
                prompt_for(row, tokenizer),
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt = {k: v[:, -args.max_length :].to(model.device) for k, v in prompt.items()}
            tokens = model.generate(
                **prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            answer = tokenizer.decode(
                tokens[0, prompt["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
            correct += answer.casefold() == str(row["output"]).strip().casefold()
            predictions.append(answer)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "exact_match": correct / len(losses),
                "mean_answer_loss": sum(losses) / len(losses),
                "predictions": predictions,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
