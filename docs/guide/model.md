# The local AI model

BrainO's agent runs on your own computer: [llama.cpp](https://github.com/ggml-org/llama.cpp)
with an open-weight Qwen model. Nothing is sent to an AI service, and there is no account or
server. The engine and its tools work without the model; you only need it for the agent.

```bash
braino model hardware   # what this computer has, and the model BrainO proposes
braino model setup      # install llama.cpp, download the model once, measure the speed
braino model status     # what is set up, and how fast it is
```

## Which model, on which computer

BrainO detects the RAM and the GPU (if there is one with its own memory) and proposes a tier.
Every tier works; larger ones answer better or faster.

| Tier | Computer | Model | Download |
|---|---|---|---|
| `minimum` | no GPU, 16 GB RAM | Qwen3.5 4B, on the CPU | 2.6 GB |
| `more-ram` | no GPU, 32 GB RAM or more | Qwen3.6 35B-A3B, on the CPU | 19 GB |
| `gpu` | a GPU with 6 GB or more, or a Mac with Apple silicon | the strongest model that fits: Qwen3.5 9B on an 8 GB GPU, or the 35B-A3B split between GPU and RAM | 2.6–19 GB |

The 35B-A3B is a *mixture of experts*: it has 35 billion parameters but uses only about 3 billion
for each word, so it runs about as fast as the 4B model while planning better. It needs the
RAM to hold all of it, though. On a computer with a mid-sized GPU, its shared layers go on the
GPU and its experts stay in RAM; llama.cpp works out the split from the memory available.

BrainO keeps 8 GB of RAM free for the operating system, the app and your analyses. Below
16 GB of RAM, it still sets up the 4B model but warns you that it may not fit next to your
work.

To choose differently:

```bash
braino model list                          # every model, its size, and whether it fits here
braino model setup --model qwen3.5-4b      # a smaller model, for speed
braino model setup --backend cpu           # ignore the GPU
```

The `--backend` options are `cpu`, `vulkan` (any GPU: NVIDIA, AMD, Intel), `cuda` (NVIDIA,
fastest there; a larger download, about 650 MB on Windows) and `metal` (Apple silicon, chosen
automatically). If a GPU build does not start on your computer, `--backend vulkan` or
`--backend cpu` will.

## Downloads

Everything is downloaded once, into BrainO's data folder (`~/.cache/braino`, or the folder in
`BRAINO_DATA`; set it to use a drive with more space).

- **Pinned:** the llama.cpp build and every model file are pinned by SHA-256 (models also by
  their exact repository revision), so you get exactly the files BrainO was tested with. A file
  that does not match is deleted, not used.
- **Resumable:** an interrupted download continues where it stopped the next time you run
  `braino model setup`; dropped connections are retried.
- **Licences:** the models are under Apache-2.0 and llama.cpp under MIT. Both are downloaded
  from their publishers (Hugging Face, GitHub), not bundled with BrainO.

`braino model remove qwen3.6-35b-a3b` deletes a model; `braino model remove --runtime cuda` an
installed llama.cpp build.

## How fast it will be

After setup, a short benchmark reads a prompt made of BrainO's own tool descriptions (what the
agent reads at each step) and writes a short answer. From the measured speeds it estimates how
long an agent step takes:

```text
Reads <R> tokens/s and writes <W> tokens/s. A request's first step takes about <F> s and each
later step about <L> s (model loaded in <T> s). The agent is usable on this computer; expect
short waits.
```

The verdict is *quick* when a first step takes up to 15 s, *usable* up to 60 s, and *slow*
beyond that.

On a CPU, reading the prompt is the slow part. That is why later steps are faster: the server
keeps what it has already read, so only the new part of the conversation (a tool result) is
read again. Run `braino model bench` again after changing the model or the hardware.

## The server

The agent starts and stops the model server itself; you do not run it by hand. It listens on
`127.0.0.1` only, on a free port, with a new random API key each time, so other programs and
other users of the computer cannot use it. Its log is `runtime/logs/llama-server.log` in the
data folder.

For development, `braino model serve` runs it in the foreground and prints its
OpenAI-compatible address and key. `BRAINO_LLAMA_SERVER=/path/to/llama-server` makes BrainO use
a llama.cpp you built yourself (for example on an HPC cluster) instead of the downloaded one.

## How the agent talks to the model

The agent works one step at a time. Each step the model writes one JSON object: call a tool,
load the description of more tools, ask the user to approve a proposal, ask the user a
question, or give the final answer.

- **No invalid tool calls.** Each reply must match a JSON schema built from the tools the
  model has loaded, with each tool's own parameters. llama.cpp turns the schema into a grammar,
  so the model cannot call a tool that does not exist, misspell a parameter or pass a value of
  the wrong type. (The API's native tool calls are not used for this: llama.cpp does not check
  their parameters against the schema for Qwen's tool-call format.) Checks a schema cannot
  express, such as `tmin` before `tmax`, are made when the tool runs, and the error goes back
  to the model.
- **Only the tools a step needs.** All of BrainO's tool descriptions together are about
  6,500 tokens, close to two minutes of reading for the 4B model on a CPU. The model sees a
  one-line list of every tool, starts with the few the request's words point to, and loads
  others when it needs them (or simply calls one, which loads it). A tool named in a result, as in "run eeg.preprocess first", is
  loaded automatically.
- **The prompt cache is reused.** The conversation only grows at the end, and the instructions
  at its start are the same for every request, so llama.cpp keeps them cached. A later step
  reads only what is new: the last tool result, and any tool descriptions just loaded.
- **Reasoning is off.** Qwen3.5 can think step by step before it replies, but on a CPU that
  costs about 20 s of writing per step. The agent turns it off; the step-by-step structure
  does that work instead.

The client speaks the OpenAI-compatible API, so another server that supports JSON-schema
replies (Ollama, vLLM) can stand in for llama.cpp:

```python
from braino.llm import ChatClient, Conversation

client = ChatClient(
    "http://localhost:11434/v1",
    model="my-qwen",  # the model's name on that server
    llama_cpp=False,
)
conversation = Conversation(client, "Check the data quality of subject 02.")
step = conversation.next_step()  # e.g. CallTool(tool="eeg.qc_report", arguments={...})
```

From Python:

```python
from braino.runtime import describe, detect, recommend, setup, start_server

print(recommend(detect()).model)  # the proposed model
settings = setup()  # install, download, benchmark
print(describe(settings.benchmark))

with start_server() as server:
    reply = server.post(
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32},
    )
```
