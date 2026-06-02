import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--run-name", default="v1")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--data-dir", type=Path,
                   default=REPO_ROOT / "data" / "processed")
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "runs")
    args = p.parse_args()

    out_dir = args.out_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading datasets from {args.data_dir}...")
    train_ds = load_jsonl(args.data_dir / "train.jsonl")
    val_ds = load_jsonl(args.data_dir / "val.jsonl")
    print(f"  train: {len(train_ds)}, val: {len(val_ds)}")

    print(f"loading tokenizer + model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=args.max_len,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        dataset_kwargs={"skip_prepare_dataset": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora,
        processing_class=tokenizer,
    )

    print("=" * 60)
    print(f"trainable params:")
    trainer.model.print_trainable_parameters()
    print("=" * 60)
    print("starting training...")
    trainer.train()

    print(f"\nsaving final adapter to {out_dir}/final/")
    trainer.save_model(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))
    print("done")


if __name__ == "__main__":
    main()
