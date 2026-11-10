"""Fine-tune the agent's model with LoRA on ``braino train data`` examples.

    python train_lora.py --data data/ --out runs/pilot

Loads the base model (Qwen3.5 4B at a pinned revision), adds LoRA adapters to the language
model's linear layers (attention, linear attention and MLP; not the vision tower), and trains
on the steps the agent writes (see render.py). Saves the adapter, the settings and the
validation loss of each epoch. ``merge.py`` then folds the adapter into the model for GGUF
conversion.

Precision follows the GPU: bf16 where supported (RTX 30xx and later), else fp16 with loss
scaling (the HPC's V100s). ``--load-4bit`` loads the base in 4 bits (QLoRA) to fit a smaller
GPU; by default it is used below 20 GB of GPU memory.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from render import IGNORE, read_jsonl, render
from transformers import AutoModelForImageTextToText, AutoTokenizer

BASE = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
#: LoRA on the language model's decoder layers: full attention (q, k, v, o), linear attention
#: (the qkv and gate inputs and the output; not the tiny per-head a and b inputs) and the MLP.
#: The vision tower (model.visual), embeddings and output head stay as they are.
TARGETS = (
    r".*language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|"
    r"linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)$"
)


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, required=True, help="Folder with train.jsonl, valid.jsonl")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base", default=BASE)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--accumulate", type=int, default=8, help="Examples per optimiser step")
    p.add_argument("--max-tokens", type=int, default=12288, help="Longer examples are skipped")
    p.add_argument("--warmup", type=float, default=0.05, help="Share of steps warming up")
    p.add_argument("--load-4bit", choices=["auto", "yes", "no"], default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="Use only this many examples (a test)")
    return p.parse_args()


def load_examples(tokenizer: Any, path: Path, max_tokens: int, limit: int | None) -> list[Any]:
    out, skipped = [], 0
    for i, ex in enumerate(read_jsonl(path)):
        if limit is not None and i >= limit:
            break
        r = render(tokenizer, ex)
        if len(r.input_ids) > max_tokens:
            skipped += 1
            continue
        out.append(r)
    if skipped:
        print(f"{path.name}: skipped {skipped} examples longer than {max_tokens} tokens")
    return out


def load_model(args: argparse.Namespace, dtype: torch.dtype) -> Any:
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    four_bit = args.load_4bit == "yes" or (args.load_4bit == "auto" and memory_gb < 20)
    kwargs: dict[str, Any] = {"revision": args.revision, "dtype": dtype}
    if four_bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForImageTextToText.from_pretrained(args.base, device_map={"": 0}, **kwargs)
    print(f"GPU {torch.cuda.get_device_name(0)} ({memory_gb:.0f} GB); 4-bit base: {four_bit}")
    if four_bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False
    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=TARGETS,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, four_bit


def loss_of(model: Any, r: Any) -> torch.Tensor:
    ids = torch.tensor([r.input_ids], device="cuda")
    labels = torch.tensor([r.labels], device="cuda")
    return model(input_ids=ids, labels=labels).loss


@torch.no_grad()
def validate(model: Any, examples: list[Any]) -> float:
    model.eval()
    total, tokens = 0.0, 0
    for r in examples:
        n = sum(1 for x in r.labels if x != IGNORE)
        total += loss_of(model, r).item() * n
        tokens += n
    model.train()
    return total / max(tokens, 1)


def main() -> None:
    args = parse()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.base, revision=args.revision)
    train = load_examples(tokenizer, args.data / "train.jsonl", args.max_tokens, args.limit)
    valid_path = args.data / "valid.jsonl"
    valid = (
        load_examples(tokenizer, valid_path, args.max_tokens, args.limit)
        if valid_path.exists()
        else []
    )
    print(f"{len(train)} training and {len(valid)} validation examples; dtype {dtype}")

    model, four_bit = load_model(args, dtype)
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total_steps = math.ceil(len(train) * args.epochs / args.accumulate)
    warmup = max(1, int(total_steps * args.warmup))

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_at)
    scaler = torch.amp.GradScaler("cuda", enabled=not bf16 and not four_bit)
    history: list[dict[str, Any]] = []
    if valid:
        history.append({"epoch": 0, "valid_loss": validate(model, valid)})
        print(f"before training: validation loss {history[-1]['valid_loss']:.4f}")

    model.train()
    step, seen, clock = 0, 0, time.perf_counter()
    order: list[int] = []
    budget = int(len(train) * args.epochs)
    running = 0.0
    while seen < budget:
        if not order:
            order = list(range(len(train)))
            random.shuffle(order)
        r = train[order.pop()]
        with torch.autocast("cuda", dtype=dtype):
            loss = loss_of(model, r) / args.accumulate
        scaler.scale(loss).backward()
        running += loss.item()
        seen += 1
        if seen % args.accumulate == 0 or seen == budget:
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            if step % 5 == 0 or step == total_steps:
                rate = seen / (time.perf_counter() - clock)
                print(
                    f"step {step}/{total_steps}  loss {running:.4f}  "
                    f"lr {scheduler.get_last_lr()[0]:.2e}  {rate:.2f} examples/s"
                )
            running = 0.0
        if valid and seen % len(train) == 0:
            epoch = seen // len(train)
            history.append({"epoch": epoch, "valid_loss": validate(model, valid)})
            print(f"epoch {epoch}: validation loss {history[-1]['valid_loss']:.4f}")
            model.save_pretrained(args.out / f"epoch-{epoch}")

    model.save_pretrained(args.out / "adapter")
    tokenizer.save_pretrained(args.out / "adapter")
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    summary = {
        "settings": settings,
        "examples": {"train": len(train), "valid": len(valid)},
        "steps": step,
        "dtype": str(dtype),
        "four_bit": four_bit,
        "minutes": round((time.perf_counter() - clock) / 60, 1),
        "history": history,
    }
    (args.out / "training.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"adapter saved to {args.out / 'adapter'}")


if __name__ == "__main__":
    main()
