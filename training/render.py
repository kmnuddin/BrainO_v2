"""Turn training examples into token ids and labels, exactly as llama.cpp renders the prompt.

Each example (a line of ``braino train data``'s JSONL) is one conversation: the system prompt,
the request, the agent's steps and the tool results. It is rendered with the model's own chat
template, with thinking off as BrainO runs it. Qwen3.5's template renders every step after
the request as ``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n{step}<|im_end|>``, which
is exactly the generation prompt BrainO sends plus what the model writes. So one rendering
of the whole conversation serves every step, and the loss covers only the steps the model
wrote (weight 1): the JSON and the ``<|im_end|>`` after it.

Needs ``transformers`` (and ``jinja2``), not torch.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: What the template puts between the assistant header and a step when thinking is off.
STEP_PREFIX = "<think>\n\n</think>\n\n"
END = "<|im_end|>"
IGNORE = -100


@dataclass
class Rendered:
    input_ids: list[int]
    labels: list[int]
    text: str

    @property
    def trained_tokens(self) -> int:
        return sum(1 for x in self.labels if x != IGNORE)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def render(tokenizer: Any, example: dict[str, Any]) -> Rendered:
    """Token ids and labels of one example; labels are -100 outside the trained steps."""
    messages = [{"role": m["role"], "content": m["content"]} for m in example["messages"]]
    text = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
    spans = []
    for i, m in enumerate(example["messages"]):
        if m["role"] != "assistant" or m.get("weight", 1) == 0:
            continue
        before = tokenizer.apply_chat_template(
            messages[:i], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        if not before.endswith(STEP_PREFIX):
            raise ValueError("the chat template does not end the prompt with an empty think block")
        if not text.startswith(before):
            raise ValueError(f"message {i}: the conversation does not render as its prompt")
        start = len(before)
        step = m["content"] + END
        if text[start : start + len(step)] != step:
            raise ValueError(f"message {i}: the step is not where the prompt ends")
        spans.append((start, start + len(step)))
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids: list[int] = encoded["input_ids"]
    labels = [IGNORE] * len(ids)
    for k, (a, b) in enumerate(encoded["offset_mapping"]):
        if any(a >= s and b <= e and b > a for s, e in spans):
            labels[k] = ids[k]
    return Rendered(input_ids=ids, labels=labels, text=text)


def check(tokenizer: Any, path: Path, limit: int | None = None) -> None:
    """Render the examples of ``path`` and print their sizes; shows one trained span."""
    lengths, trained = [], []
    shown = False
    for n, example in enumerate(read_jsonl(path)):
        if limit is not None and n >= limit:
            break
        r = render(tokenizer, example)
        lengths.append(len(r.input_ids))
        trained.append(r.trained_tokens)
        if not shown:
            pairs = zip(r.input_ids, r.labels, strict=True)
            pieces = [tokenizer.decode([i]) for i, lab in pairs if lab != IGNORE]
            print("trained text of the first example:", repr("".join(pieces)[:600]))
            shown = True
    lengths.sort()
    print(
        f"{len(lengths)} examples; tokens min {lengths[0]}, median {lengths[len(lengths) // 2]}, "
        f"max {lengths[-1]}; trained tokens {sum(trained)} of {sum(lengths)}"
    )


if __name__ == "__main__":
    import argparse

    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path, help="A JSONL file from braino train data")
    parser.add_argument("--base", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(args.base, revision=args.revision)
    check(tok, args.data, args.limit)
