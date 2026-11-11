"""Merge a LoRA adapter into the base model and save fp16 (CPU-friendly).

Usage: python merge_lora.py <adapter_dir> <output_dir>
"""
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER = sys.argv[1] if len(sys.argv) > 1 else "./adapter"
OUT = sys.argv[2] if len(sys.argv) > 2 else "./merged_model"

model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16)
tok = AutoTokenizer.from_pretrained(BASE)
model = PeftModel.from_pretrained(model, ADAPTER)
model = model.merge_and_unload()
model.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
print(f"merged -> {OUT}")
