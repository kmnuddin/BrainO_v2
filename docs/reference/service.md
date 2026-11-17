# Engine service

The desktop app talks to the engine through a small web service on your own computer. You
don't need it to use BrainO from Python or the command line. It is described here for people
working on the app, or on their own front end.

```bash
pip install "braino[server]"
BRAINO_SERVER_TOKEN=<a long random string> braino serve --exit-with-stdin
```

When it is ready, it prints one line of JSON:

```json
{"braino_server": {"url": "http://127.0.0.1:53817", "pid": 4242}}
```

- **It listens on this computer only** (127.0.0.1, or `--host localhost` / `::1`); other
  addresses are refused, because the service can read and write your files. `--port 0` (the
  default) lets the system choose a free port.
- **Every request needs the token**, except `GET /api/health`: in an `Authorization: Bearer
  <token>` or `X-BrainO-Token` header. Without `BRAINO_SERVER_TOKEN`, the service makes one
  and adds it to the ready line. Requests must also name this computer as their host, which
  stops a web page from reaching the service through DNS rebinding.
- **`--exit-with-stdin`** stops the service when its standard input closes, so it ends with
  the app that started it. Stopping the service cancels its unfinished jobs.
- **`--allow-origin`** lets a page from that origin call it (the app's own page, for
  example); none are allowed by default.

## Jobs

Quick reads answer at once. Work that takes time runs as a **job**: the request returns the job
straight away, queued, and it then runs in a process of its own. That way a job can be
cancelled, and a crash in one job cannot stop the service. A job ends `done`, `failed` or
`cancelled`.

Jobs run side by side, as many as the computer allows: one for every two logical processors,
but at most one for every 3 GB of memory after the first 6 GB (left for the system, the app
and the AI model). The processors are shared among the jobs. On the minimum computer (16 GB),
that is three jobs at once. Further jobs wait in the queue.

Two jobs that would write the same files are not allowed at once: the same pipeline on the
same dataset, two imports into one dataset, or two decisions on one proposal. The second
request gets `409`.

A job reports its progress (`progress`, from 0 to 1, and `message`), its provenance run
(`run_id`), and *events*: each pipeline step's outcome, or each imported recording. At the
end it gives its `result` or its `error`. An error is the exception's message only, never a
traceback, because tracebacks contain file paths that can identify the person recorded.

## Events

`/api/events` is a WebSocket. Browsers cannot send headers on a WebSocket, so the token goes
in as its second subprotocol:

```js
const events = new WebSocket(url.replace("http", "ws") + "/api/events", ["braino", token]);
```

It first sends `{"type": "jobs", "jobs": [...]}`, then one message for every change:

- `{"type": "job", "job": {...}}`: the job, without its events and result. When it is
  finished, `GET /api/jobs/{id}` gives them.
- `{"type": "job_event", "job_id": "...", "event": {...}}`: something the job reported, such
  as a pipeline step's outcome.

## Routes

| Route | What it does |
|---|---|
| `GET /api/health` | Whether the service is up, and its version (no token needed) |
| `GET /api/system` | The computer, how many jobs run at once, the chosen model, the model recommended for this computer, the installed models, every model's size, and your BrainO folder |
| `GET /api/tools`, `GET /api/tools/{name}` | The tools, with their risk and the JSON schemas of their input and output (the app builds its forms from these) |
| `GET /api/datasets/summary?path=` | A dataset's subjects, sessions, tasks and recordings |
| `POST /api/import/detect` | The recordings among files and folders, with their formats; other files, with the reason they were left out |
| `POST /api/jobs/import` | Import recordings (job): `dataset`, `key` (the subject key file), `jobs` (`source`, `subject_id`, `task`, `session`, `run`, `line_freq`) |
| `POST /api/jobs/tool` | Run one tool (job): `tool`, `arguments`, `dataset`. The arguments are checked before the job is queued. A decision tool's result is its pending proposal |
| `GET /api/proposals?dataset=&status=` | The dataset's proposals, optionally only `pending`, `approved` or `rejected` |
| `GET /api/proposals/{id}?dataset=` | One proposal |
| `GET /api/proposals/{id}/evidence?dataset=` | What the person needs in order to decide, read from the recording. For bad channels: the electrode layout, each channel's amplitude, and ten seconds of each proposed channel beside its nearest good neighbour. For ICA: each component's weights at the electrodes and the spectrum of its activity |
| `POST /api/proposals/{id}/decide` | Approve or reject (job): `dataset`, `approve`, optional `edits` and `note`. The decision is recorded in the name of the person using the computer |
| `GET /api/pipelines` | The built-in pipelines, then the ones you saved |
| `GET`, `PUT`, `DELETE /api/pipelines/{name}` | Read, save or delete a pipeline (`PUT` takes `{"spec": ...}`). Built-in pipelines cannot be replaced or deleted |
| `POST /api/pipelines/check` | Check a spec: its steps against the tools, then the whole plan, against the data if `dataset` is given |
| `GET /api/pipelines/{name}/state?dataset=` | For each subject and step: done, waiting for a decision, or rejected |
| `GET /api/datasets/pipelines?dataset=` | The pipelines that have run on the dataset, as they last ran, including pipelines that were never saved (from a template or the builder) |
| `GET /api/pipelines/{name}/results?dataset=` | The pipeline's results: its ERP, spectrum and time-frequency steps with the subjects that have results, the measures, and each group test's result |
| `GET /api/pipelines/{name}/results/{step}?dataset=` | One measure step's data at one `channel`, averaged over the subjects or for one `subject`. For ERPs, also each waveform's scalp map between `tmin` and `tmax`. For time-frequency, the map of one `condition` |
| `GET /api/templates` | The pipeline templates, with the JSON schema of their settings |
| `POST /api/templates/{name}/build` | A template's pipeline for a dataset's task and subjects: `settings`, `dataset` |
| `POST /api/jobs/pipeline` | Run a pipeline (job): `dataset`, a pipeline `name` or a `spec`, optional `subjects`, `from_step`, `auto_approve`. A plan that fails its check is refused with `422` and its findings |
| `POST /api/jobs/example` | Download the first `subjects` (10 by default) of an ERP CORE `task` (P3 or N170), checked against pinned hashes (job). The result names the dataset's folder |
| `POST /api/jobs/model-setup` | Install llama.cpp and download the model (job), with download progress: `model` (the recommended one by default), `backend`, `benchmark` |
| `GET /api/jobs`, `GET /api/jobs/{id}` | This session's jobs, newest first; one job in full |
| `POST /api/jobs/{id}/cancel` | Cancel a queued job, or stop a running one |
| `GET /api/runs?dataset=`, `GET /api/runs/{id}?dataset=` | The dataset's recorded runs; one run with all its tool calls |
| `WS /api/events` | Job changes as they happen (see above) |

Errors come back as `{"detail": ...}`:

- `400`: a folder that does not exist;
- `401`: a missing or wrong token;
- `404`: an unknown tool, proposal, pipeline, run or job;
- `409`: a conflict, such as a job already working on it, a decided proposal, or a built-in
  pipeline;
- `422`: arguments or a spec that do not fit;
- `501`: something that needs the `eeg` extra, which is not installed.

A pipeline's state keeps the spec it last ran with. That is how pipelines that were never
saved still have results, and how `/api/datasets/pipelines` lists them.

## Saved pipelines

Pipelines saved through the service are YAML files in your BrainO folder: `%APPDATA%\BrainO\pipelines`
on Windows, `~/Library/Application Support/BrainO/pipelines` on macOS, `~/.config/braino/pipelines`
on Linux, or `$BRAINO_HOME/pipelines`. They are in the same format as the built-in ones, so
`braino pipeline run <file>` runs them too. Clearing BrainO's download cache does not touch them.
