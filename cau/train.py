"""Train conflict-aware unlearning on a prepared forget/retain split."""

import argparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "forget-file", "retain-file", "output-dir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--module-regex", default=r"model\.layers\.7")
    p.add_argument(
        "--trainable-regex",
        nargs="+",
        default=[r"model\.layers\.(5|6|7)\.mlp\.down_proj\.weight"],
    )
    p.add_argument("--steering", type=float, default=20)
    p.add_argument("--forget-weight", type=float, default=1)
    p.add_argument("--retain-weight", type=float, default=1)
    args = p.parse_args()
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from .data import read_records, PairedDataset, PairedCollator
    from .objective import ConflictObjective, select_parameters

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define an EOS token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if use_bf16 else torch.float32
    )
    model.config.use_cache = False
    selected = select_parameters(model, args.trainable_regex)
    objective = ConflictObjective(
        model, args.module_regex, args.steering, args.forget_weight, args.retain_weight
    )

    class CAUTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            loss, outputs = objective(model, inputs)
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        remove_unused_columns=False,
        report_to="none",
        save_strategy="no",
        logging_steps=10,
        bf16=use_bf16,
        seed=args.seed,
    )
    trainer = CAUTrainer(
        model=model,
        args=training_args,
        train_dataset=PairedDataset(
            read_records(args.forget_file),
            read_records(args.retain_file),
            tokenizer,
            args.max_length,
        ),
        data_collator=PairedCollator(tokenizer.pad_token_id),
    )
    print(f"Training {len(selected)} selected parameter tensors.")
    trainer.train()
    trainer.save_model(args.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
