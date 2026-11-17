# BrainO desktop app

The desktop app: Electron with a React and TypeScript page. It starts the engine service
(`braino serve`) and works only through it. The design is in
[docs/PLAN.md](../docs/PLAN.md), section 6.1. The engine's routes are in
[docs/reference/service.md](../docs/reference/service.md).

## Developing

You need Node.js 22 or later, and the engine installed in its own virtual environment with
the `eeg` and `server` extras:

```bash
cd engine && uv venv && uv pip install -e ".[dev,eeg,server]"   # once
cd app && npm ci                                                # once
npm run dev        # the app, reloading as you edit
npm run typecheck  # TypeScript, main process and page
npm test           # unit tests (vitest)
npm run build      # the built app in out/; `npm start` runs it
```

The app finds the engine's Python in this order:

1. the `BRAINO_PYTHON` variable;
2. the Python chosen in Settings;
3. the one bundled with an installed app;
4. `engine/.venv` next to this folder;
5. the `braino` command.

When run from VS Code, unset `ELECTRON_RUN_AS_NODE`: VS Code sets it, and with it Electron
starts as plain Node.js.

## How it is put together

| Folder | What it does |
|---|---|
| `src/main/` | The main process: the window, starting and stopping the engine (`engine.ts`), the app's settings, and what the page may ask of the computer (dialogs, memory, free space) |
| `src/preload/` | The bridge, `window.braino`: the page's only way out |
| `src/shared/` | The types the main process and the page share |
| `src/renderer/src/` | The page: `engine/` (the API client and the engine's types), `state/` (the app, the open dataset, the jobs), `screens/`, `components/` and `lib/` |

The page never sees the engine's token. Requests and the WebSocket's events go through the
main process. The page runs sandboxed, with context isolation and without Node.js. The built
page loads only its own files (a content security policy is added at build time), and
navigation and new windows are refused.

## Checking screens without clicking

`BRAINO_CAPTURE` names a JSON file of steps: click, type, wait for text, take a screenshot,
answer the next file dialog. When it is set, an unpackaged app runs the steps, saves the
screenshots next to the file, and quits. See `src/main/capture.ts`. To keep a run from
touching your real settings and pipelines, set `BRAINO_APP_DATA` and `BRAINO_HOME` to folders
of their own.
