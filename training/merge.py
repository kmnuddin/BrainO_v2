"""Fold a LoRA adapter into the base model and save it for GGUF conversion.

    python merge.py --adapter runs/pilot/adapter --out runs/pilot/merged

The merged model keeps the base model's architecture (with its vision tower), so llama.cpp's
``convert_hf_to_gguf.py`` converts it exactly as it converts the base. Merging runs on the
CPU in about 10 GB of RAM. Then, from a llama.cpp checkout at the build BrainO pins:

    python convert_hf_to_gguf.py runs/pilot/merged --outtype bf16 --outfile braino-4b-bf16.gguf
    llama-quantize braino-4b-bf16.gguf braino-4b-Q4_K_M.gguf Q4_K_M
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from train_lora import BASE, REVISION
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base", default=BASE)
    p.add_argument("--revision", default=REVISION)
    args = p.parse_args()

    model = AutoModelForImageTextToText.from_pretrained(
        args.base, revision=args.revision, dtype=torch.bfloat16, device_map={"": "cpu"}
    )
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base, revision=args.revision).save_pretrained(args.out)
    # The processor files (image and video settings) are part of the architecture llama.cpp reads.
    AutoProcessor.from_pretrained(args.base, revision=args.revision).save_pretrained(args.out)
    print(f"merged model saved to {args.out}")


if __name__ == "__main__":
    main()
