#!/usr/bin/env python3
"""Apply LoRA support patch to FlagEmbedding encoder-only m3 trainer."""
import sys

BASE = "/proj/uppmax2025-2-505/xilu1878/FlagEmbedding/FlagEmbedding/finetune/embedder/encoder_only/m3"
args_path = f"{BASE}/arguments.py"
runner_path = f"{BASE}/runner.py"

# --- Patch arguments.py ---
with open(args_path) as f:
    content = f.read()

if 'use_lora' in content:
    print("arguments.py: already patched, skipping")
else:
    old = '    self_distill_start_step: int = field(default=-1, metadata={"help": "Num of step when using self-distill"})\n'
    addition = (
        '    use_lora: bool = field(default=False, metadata={"help": "Use LoRA for parameter-efficient fine-tuning"})\n'
        '    lora_rank: int = field(default=16, metadata={"help": "LoRA rank r"})\n'
        '    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha scaling factor"})\n'
        '    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})\n'
        '    lora_target_modules: str = field(default="query,key,value", metadata={"help": "Comma-separated modules to apply LoRA"})\n'
    )
    if old not in content:
        print("arguments.py: ERROR — could not find anchor line, aborting")
        sys.exit(1)
    content = content.replace(old, old + addition)
    with open(args_path, 'w') as f:
        f.write(content)
    print("arguments.py: patched OK")

# --- Patch runner.py ---
with open(runner_path) as f:
    content = f.read()

if 'use_lora' in content:
    print("runner.py: already patched, skipping")
else:
    old_block = (
        '        model = EncoderOnlyEmbedderM3Model(\n'
        '            self.get_model(self.model_args.model_name_or_path, self.model_args.trust_remote_code, self.model_args.colbert_dim),\n'
        '            tokenizer=tokenizer,\n'
    )
    new_block = (
        '        model_dict = self.get_model(\n'
        '            self.model_args.model_name_or_path,\n'
        '            self.model_args.trust_remote_code,\n'
        '            self.model_args.colbert_dim\n'
        '        )\n'
        '\n'
        '        if self.training_args.use_lora:\n'
        '            from peft import get_peft_model, LoraConfig\n'
        '            lora_config = LoraConfig(\n'
        '                r=self.training_args.lora_rank,\n'
        '                lora_alpha=self.training_args.lora_alpha,\n'
        '                target_modules=[m.strip() for m in self.training_args.lora_target_modules.split(",")],\n'
        '                lora_dropout=self.training_args.lora_dropout,\n'
        '                bias="none",\n'
        '            )\n'
        '            model_dict[\'model\'] = get_peft_model(model_dict[\'model\'], lora_config)\n'
        '            model_dict[\'model\'].enable_input_require_grads()\n'
        '            model_dict[\'model\'].print_trainable_parameters()\n'
        '\n'
        '        model = EncoderOnlyEmbedderM3Model(\n'
        '            model_dict,\n'
        '            tokenizer=tokenizer,\n'
    )
    if old_block not in content:
        print("runner.py: ERROR — could not find anchor block, aborting")
        sys.exit(1)
    content = content.replace(old_block, new_block)
    with open(runner_path, 'w') as f:
        f.write(content)
    print("runner.py: patched OK")

print("Done.")
