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
from torch.utils.checkpoint import checkpoint
from transformers import (
    AttentionInterface,
    AttentionMaskInterface,
    AutoModelForImageTextToText,
    AutoTokenizer,
)
from transformers.masking_utils import sdpa_mask

BASE = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
#: LoRA on the language model's decoder layers: full attention (q, k, v, o), linear attention
#: (the qkv and gate inputs and the output; not the tiny per-head a and b inputs) and the MLP.
#: The vision tower (model.visual), embeddings and output head stay as they are.
TARGETS = (
    r".*language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|"
    r"linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)$"
)


def sdpa_expanded(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **_: Any,
) -> tuple[torch.Tensor, None]:
    """SDPA attention that never builds the whole attention matrix.

    Without a padding mask, transformers' "sdpa" lets SDPA expand the key heads itself
    (``enable_gqa``). The memory-efficient kernel refuses that, and where flash attention is
    missing (Windows builds) SDPA falls back to the math kernel, which holds the full matrix:
    about 6 GB per layer at 10k tokens. Here the key heads are expanded first.
    """
    groups = query.shape[1] // key.shape[1]
    if groups > 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    if attention_mask is not None:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
    causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    out = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout if module.training else 0.0,
        scale=scaling,
        is_causal=bool(causal and attention_mask is None and query.shape[2] > 1),
    )
    return out.transpose(1, 2).contiguous(), None


ATTENTION = "sdpa_expanded"
AttentionInterface.register(ATTENTION, sdpa_expanded)
AttentionMaskInterface.register(ATTENTION, sdpa_mask)


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
    p.add_argument(
        "--memory-fraction", type=float, default=0.85, help="Share of GPU memory PyTorch may use"
    )
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
    kwargs: dict[str, Any] = {
        "revision": args.revision,
        "dtype": dtype,
        "attn_implementation": ATTENTION,
    }
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
        # That casts every weight it did not quantise to fp32, the embedding table (shared with
        # the output head) too: 2.5 GB instead of 1.3, and fp32 hidden states, which gradient
        # checkpointing keeps for every layer. The table is frozen; bf16 is enough.
        model.get_input_embeddings().to(dtype)
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


LOSS_CHUNK = 256
"""Trained positions whose logits are computed at once."""


def loss_of(model: Any, r: Any) -> torch.Tensor:
    """Mean cross-entropy over the trained tokens.

    Logits are computed only at the positions that predict a trained token, a chunk at a time,
    and computed again in the backward pass instead of kept: at Qwen's 248k-token vocabulary
    they take 1 GB per 1,000 positions in fp32, and a long example trains on a few thousand.
    """
    ids = torch.tensor([r.input_ids], device="cuda")
    labels = torch.tensor(r.labels, device="cuda")
    keep = torch.nonzero(labels[1:] != IGNORE).squeeze(-1)  # position t predicts token t + 1
    inner = model.get_base_model()  # LoRA layers are inside it
    hidden = inner.model(input_ids=ids).last_hidden_state[0, keep]
    targets = labels[keep + 1]
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, len(keep), LOSS_CHUNK):
        part = slice(start, start + LOSS_CHUNK)
        total = total + checkpoint(
            _loss_sum, hidden[part], targets[part], inner.lm_head.weight, use_reentrant=False
        )
    return total / len(keep)


def _loss_sum(hidden: torch.Tensor, targets: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
    logits = torch.nn.functional.linear(hidden, head).float()
    return torch.nn.functional.cross_entropy(logits, targets, reduction="sum")


@torch.no_grad()
def validate(model: Any, examples: list[Any]) -> float:
    model.eval()
    total, tokens = 0.0, 0
    for r in examples:
        torch.cuda.empty_cache()
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
    # Stay in GPU memory: past the cap, PyTorch frees its cached blocks (fragmented by examples
    # of every length) and retries, where Windows drivers would spill into system RAM and make
    # training several times slower without an error. The cap leaves room for the CUDA context
    # and, on a desktop, the display.
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
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
    skipped: list[int] = []
    while seen < budget:
        if not order:
            order = list(range(len(train)))
            random.shuffle(order)
        r = train[order.pop()]
        # Each example starts with only the weights, adapter and optimiser state in memory:
        # cached blocks left by examples of other lengths fragment the GPU's memory.
        torch.cuda.empty_cache()
        try:
            with torch.autocast("cuda", dtype=dtype):
                loss = loss_of(model, r) / args.accumulate
            scaler.scale(loss).backward()
            running += loss.item()
        except torch.OutOfMemoryError:
            # Gradients from the layers it reached stay in the step: a small error, rather than
            # losing the run.
            skipped.append(len(r.input_ids))
            print(f"out of memory on an example of {len(r.input_ids)} tokens: skipped", flush=True)
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
                    f"lr {scheduler.get_last_lr()[0]:.2e}  {rate:.2f} examples/s  "
                    f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB",
                    flush=True,
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
        "skipped_out_of_memory": skipped,
        "dtype": str(dtype),
        "four_bit": four_bit,
        "minutes": round((time.perf_counter() - clock) / 60, 1),
        "history": history,
    }
    (args.out / "training.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"adapter saved to {args.out / 'adapter'}")


if __name__ == "__main__":
    main()
