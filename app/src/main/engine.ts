/** Starts the engine service (`braino serve`), talks to it, and stops it with the app.
 *
 * The engine listens on 127.0.0.1 with a token made here for this session. The token stays in
 * the main process: the page asks through IPC, and events from the engine's WebSocket are
 * passed on to it. */

import { type ChildProcess, spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { EventEmitter } from "node:events";
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { createInterface } from "node:readline";
import type { ApiResponse, EngineStatus, HttpMethod } from "../shared/ipc";

const LOG_LINES = 60;
const STOP_GRACE_MS = 5000;
const READY_TIMEOUT_MS = 90_000;

export interface ReadyLine {
  url: string;
  pid: number;
}

/** The engine's ready line, `{"braino_server": {"url": ..., "pid": ...}}`, or null. */
export function parseReadyLine(line: string): ReadyLine | null {
  const text = line.trim();
  if (!text.startsWith("{")) return null;
  try {
    const value = JSON.parse(text) as { braino_server?: { url?: unknown; pid?: unknown } };
    const ready = value.braino_server;
    if (!ready || typeof ready.url !== "string") return null;
    const url = new URL(ready.url);
    if (!["127.0.0.1", "localhost", "[::1]"].includes(url.hostname)) return null;
    return { url: ready.url, pid: typeof ready.pid === "number" ? ready.pid : -1 };
  } catch {
    return null;
  }
}

/** The Python that runs the engine, with the arguments before `serve`.
 *
 * In order: the BRAINO_PYTHON variable, the Settings screen's choice, the Python bundled with
 * an installed app, the engine's own virtual environment next to the app's source (when
 * developing), and last the `braino` command on the PATH. */
export function engineCommand(options: {
  configured: string;
  appPath: string;
  resourcesPath: string;
  packaged: boolean;
  env: NodeJS.ProcessEnv;
  platform: NodeJS.Platform;
  exists?: (path: string) => boolean;
}): string[] {
  const exists = options.exists ?? existsSync;
  const python = (root: string) =>
    options.platform === "win32" ? join(root, "python.exe") : join(root, "bin", "python");
  const venvPython = (root: string) =>
    options.platform === "win32" ? join(root, "Scripts", "python.exe") : join(root, "bin", "python");
  const chosen = options.env.BRAINO_PYTHON || options.configured;
  if (chosen) return [chosen, "-m", "braino"];
  const candidates = options.packaged
    ? [python(join(options.resourcesPath, "engine"))]
    : [venvPython(resolve(options.appPath, "..", "engine", ".venv"))];
  for (const candidate of candidates) {
    if (exists(candidate)) return [candidate, "-m", "braino"];
  }
  return ["braino"];
}

export class Engine extends EventEmitter {
  private process: ChildProcess | null = null;
  private socket: WebSocket | null = null;
  private url: string | null = null;
  private readonly token = randomBytes(24).toString("hex");
  private status: EngineStatus = { state: "stopped", log: [] };
  private stopping = false;
  private retryMs = 1000;

  constructor(private readonly command: () => string[]) {
    super();
  }

  getStatus(): EngineStatus {
    return { ...this.status, log: [...this.status.log] };
  }

  start(): void {
    if (this.process) return;
    const [exe, ...prefix] = this.command();
    const args = [...prefix, "serve", "--exit-with-stdin"];
    this.stopping = false;
    this.url = null;
    this.setStatus({ state: "starting", command: [exe, ...args], log: [] });
    let child: ChildProcess;
    try {
      child = spawn(exe, args, {
        env: { ...process.env, BRAINO_SERVER_TOKEN: this.token, PYTHONIOENCODING: "utf-8" },
        stdio: ["pipe", "pipe", "pipe"],
        windowsHide: true,
      });
    } catch (error) {
      this.setStatus({ state: "failed", error: `could not start the engine: ${String(error)}` });
      return;
    }
    this.process = child;
    const timer = setTimeout(() => {
      if (this.status.state === "starting") {
        this.setStatus({ state: "failed", error: "the engine did not start within 90 seconds" });
        child.kill();
      }
    }, READY_TIMEOUT_MS);

    createInterface({ input: child.stdout! }).on("line", (line) => {
      const ready = parseReadyLine(line);
      if (ready && this.status.state === "starting") {
        clearTimeout(timer);
        this.url = ready.url;
        this.setStatus({ state: "running", error: undefined });
        this.connectEvents();
      }
    });
    createInterface({ input: child.stderr! }).on("line", (line) => this.note(line));
    child.on("error", (error) => {
      clearTimeout(timer);
      this.process = null;
      const missing = (error as NodeJS.ErrnoException).code === "ENOENT";
      this.setStatus({
        state: "failed",
        error: missing
          ? `${exe} was not found. Install BrainO (pip install "braino[eeg,server]") or choose its Python in Settings.`
          : `could not start the engine: ${error.message}`,
      });
    });
    child.on("exit", (code) => {
      clearTimeout(timer);
      this.process = null;
      this.url = null;
      this.socket?.close();
      this.socket = null;
      if (this.stopping) {
        this.setStatus({ state: "stopped" });
      } else if (this.status.state !== "failed") {
        const last = this.status.log.at(-1);
        this.setStatus({
          state: "failed",
          error: `the engine stopped (exit code ${code})` + (last ? `: ${last}` : ""),
        });
      }
    });
  }

  async stop(): Promise<void> {
    const child = this.process;
    if (!child) return;
    this.stopping = true;
    child.stdin?.end(); // --exit-with-stdin: the engine cancels its jobs and exits
    await new Promise<void>((done) => {
      const timer = setTimeout(() => {
        child.kill();
        done();
      }, STOP_GRACE_MS);
      child.once("exit", () => {
        clearTimeout(timer);
        done();
      });
    });
  }

  async restart(): Promise<EngineStatus> {
    await this.stop();
    this.start();
    return this.getStatus();
  }

  async request<T>(method: HttpMethod, path: string, body?: unknown): Promise<ApiResponse<T>> {
    if (!path.startsWith("/api/")) throw new Error(`not an engine route: ${path}`);
    if (!this.url) {
      return { ok: false, status: 503, data: { detail: "the engine is not running" } as T };
    }
    const response = await fetch(this.url + path, {
      method,
      headers: {
        Authorization: `Bearer ${this.token}`,
        ...(body === undefined ? {} : { "Content-Type": "application/json" }),
      },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const text = await response.text();
    let data: unknown = text;
    try {
      data = text ? JSON.parse(text) : null;
    } catch {
      // not JSON: keep the text
    }
    return { ok: response.ok, status: response.status, data: data as T };
  }

  private connectEvents(): void {
    if (!this.url) return;
    let socket: WebSocket;
    try {
      socket = new WebSocket(this.url.replace(/^http/, "ws") + "/api/events", ["braino", this.token]);
    } catch (error) {
      this.note(`[app] cannot follow the engine's events: ${String(error)}`);
      return;
    }
    this.socket = socket;
    socket.addEventListener("open", () => {
      this.retryMs = 1000;
    });
    socket.addEventListener("message", (event) => {
      try {
        this.emit("event", JSON.parse(String(event.data)));
      } catch {
        // ignore what is not JSON
      }
    });
    socket.addEventListener("close", (event) => {
      if (this.socket !== socket) return;
      this.note(`[app] events connection closed (${event.code})`);
      this.socket = null;
      // Reconnect while the engine runs (e.g. after the computer slept), waiting longer each
      // time it fails, up to 15 seconds.
      if (this.status.state === "running") {
        setTimeout(() => this.connectEvents(), this.retryMs);
        this.retryMs = Math.min(this.retryMs * 2, 15_000);
      }
    });
  }

  /** A line in the engine's log (the app's own lines start with "[app]"). */
  private note(line: string): void {
    this.status = { ...this.status, log: [...this.status.log, line].slice(-LOG_LINES) };
  }

  private setStatus(changes: Partial<EngineStatus>): void {
    this.status = { ...this.status, ...changes };
    this.emit("status", this.getStatus());
  }
}
