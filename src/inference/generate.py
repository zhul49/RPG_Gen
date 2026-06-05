import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# copy of Llama-3.1-8B-Instruct
DEFAULT_BASE = "NousResearch/Meta-Llama-3.1-8B-Instruct"


def _base_for_adapter(adapter_dir):
    # each adapter records the base model it was trained on
    cfg = Path(adapter_dir) / "adapter_config.json"
    if not cfg.exists():
        return DEFAULT_BASE
    # open the config file and read out the base model name
    data = json.loads(cfg.read_text())
    base = data.get("base_model_name_or_path")
    if base:
        return base
    return DEFAULT_BASE


def load(adapter_dir, base_model=None):
    if base_model is None:
        base_model = _base_for_adapter(adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
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
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    # stick our trained adapter on top of the base model
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate(model, tokenizer, messages, max_new_tokens,
             temperature, repetition_penalty, top_p=0.9):
    # turn the messages into the raw prompt string the model expects
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    full = tokenizer.decode(out[0], skip_special_tokens=False)
    # the model echoes the whole prompt back
    marker = "<|start_header_id|>assistant<|end_header_id|>"
    response = full.split(marker)[-1]
    # strip the leftover special tokens from the end
    response = response.replace("<|eot_id|>", "")
    response = response.replace("<|end_of_text|>", "")
    return response.strip(), prompt
