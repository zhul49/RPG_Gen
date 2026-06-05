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

MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
RUN_NAME = "v12"
EPOCHS = 3
BATCH = 4
GRAD_ACCUM = 4
LR = 2e-4
LORA_R = 16
LORA_ALPHA = 32
MAX_LEN = 4096
DATA_DIR = REPO_ROOT / "data" / "processed"
OUT_DIR = REPO_ROOT / "runs"


def load_jsonl(path):
    # read a jsonl file into a dataset, one row per line
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def main():
    out_dir = OUT_DIR / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading datasets from {DATA_DIR}...")
    train_ds = load_jsonl(DATA_DIR / "train.jsonl")
    val_ds = load_jsonl(DATA_DIR / "val.jsonl")
    print(f"  train: {len(train_ds)}, val: {len(val_ds)}")

    print(f"loading tokenizer + model: {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # Llama has no pad token by default, so reuse the end-of-sequence one
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # load the base model in 4-bit to save memory
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # caching is for generation, we are training so turn it off
    model.config.use_cache = False

    # LoRA trains a few small adapter layers instead of the whole model
    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # all the training settings
    cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=MAX_LEN,
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

    # print how many weights we are actually training
    print("trainable params:")
    trainer.model.print_trainable_parameters()
    print("starting training...")
    trainer.train()

    # save the trained adapter and its tokenizer
    print(f"\nsaving final adapter to {out_dir}/final/")
    trainer.save_model(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))
    print("done")


if __name__ == "__main__":
    main()
