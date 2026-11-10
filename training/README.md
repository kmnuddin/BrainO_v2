# Fine-tuning the agent's model

For development only: users download a finished model and never train anything.

The base is Qwen3.5 4B (`Qwen/Qwen3.5-4B`, pinned revision in `train_lora.py`), the model the
minimum tier runs on the CPU. Fine-tuning teaches it BrainO's tools and conventions where the
evaluation shows gaps; the evaluation tasks are never trained on.

## 1. Training data

Made by the engine, so it always matches the agent's current prompt, tools and step format:

```bash
braino train families                                # what kinds of episodes there are
braino train data -n 1000 --seed 1 -o data/          # about 25 minutes
```

Each episode is a request with a scripted way to handle it, played through the real agent loop
by a teacher in place of the model: tools run on synthetic datasets, a scripted user answers
questions and decides on proposals, and the answer is written from the actual results. Only
clean episodes are kept (checks pass, every number in the answer comes from a tool, no error
or skipped call). `data/report.json` lists what was refused and why.

**When the system prompt, a tool's description or schema, or the step format changes, make
the data again and retrain**: the examples contain them verbatim.

`train.jsonl` and `valid.jsonl` hold one conversation per line, with `weight: 0` on assistant
messages the model did not write (tool loads added from the request's words).

## 2. Training

```bash
pip install -r requirements.txt       # after installing PyTorch for your GPU
python render.py data/train.jsonl     # check: token counts and what the loss covers
python train_lora.py --data data/ --out runs/r1
```

- LoRA (rank 16) on the language model's attention, linear-attention and MLP layers; the
  vision tower is left alone. The loss covers only the agent's steps (render.py).
- bf16 on RTX 30xx and later; fp16 with loss scaling on V100 (no bf16). Below 20 GB of GPU
  memory the base is loaded in 4 bits (QLoRA).
- Qwen3.5 mixes full attention with linear attention (Gated DeltaNet). Without the
  `flash-linear-attention` kernels (Linux), transformers falls back to a slower PyTorch
  implementation; training still works.

On the HPC, `hpc/train.slurm` runs the same steps as a batch job.

## 3. Merge, convert, quantise

```bash
python merge.py --adapter runs/r1/adapter --out runs/r1/merged
# from a llama.cpp checkout at the build BrainO pins (braino/runtime/catalog.json):
python convert_hf_to_gguf.py runs/r1/merged --outtype bf16 --outfile braino-4b-r1-bf16.gguf
llama-quantize braino-4b-r1-bf16.gguf braino-4b-r1-Q4_K_M.gguf Q4_K_M
```

## 4. Is it better?

Serve the GGUF and run the evaluation on it, three times through the wordings, on a GPU
and on the CPU (the minimum tier), and compare with the base model's reports:

```bash
llama-server -m braino-4b-r1-Q4_K_M.gguf --port 8080 --ctx-size 16384 --jinja
braino eval run --agent llm --server http://127.0.0.1:8080/v1 --repeat 3 -o r1.json
braino eval compare base.json r1.json
```

A fine-tuned model is published (and added to the catalog with its SHA-256) only if it passes
more tasks than the base and breaks none.
