/** What the main process and the page say to each other (through the preload bridge). */

export type EngineState = "starting" | "running" | "stopped" | "failed";

export interface EngineStatus {
  state: EngineState;
  /** Why the engine failed or stopped, in words. */
  error?: string;
  /** The command that started it, for the Settings screen. */
  command?: string[];
  /** The last lines the engine wrote to its error output. */
  log: string[];
}

export interface ApiResponse<T = unknown> {
  ok: boolean;
  status: number;
  data: T;
}

export type HttpMethod = "GET" | "POST" | "PUT" | "DELETE";

export interface RecentDataset {
  path: string;
  openedAt: string;
}

export interface AppSettings {
  /** Where new datasets are made. */
  datasetsFolder: string;
  firstRunDone: boolean;
  recent: RecentDataset[];
  /** Python with BrainO installed; empty to find it. */
  enginePython: string;
  agentPanelOpen: boolean;
}

export interface MemoryInfo {
  totalBytes: number;
  freeBytes: number;
}

export interface OpenDialogOptions {
  title?: string;
  kind: "file" | "files" | "folder" | "folders" | "save";
  defaultPath?: string;
  filters?: { name: string; extensions: string[] }[];
}

export interface BrainoBridge {
  engineStatus(): Promise<EngineStatus>;
  restartEngine(): Promise<EngineStatus>;
  onEngineStatus(listener: (status: EngineStatus) => void): () => void;
  /** A request to the engine service; the path starts with /api/. */
  request<T = unknown>(method: HttpMethod, path: string, body?: unknown): Promise<ApiResponse<T>>;
  /** Messages from the engine's /api/events WebSocket. */
  onEvent(listener: (message: unknown) => void): () => void;
  settings(): Promise<AppSettings>;
  updateSettings(changes: Partial<AppSettings>): Promise<AppSettings>;
  /** Puts a dataset first among the recent ones. */
  openedDataset(path: string): Promise<AppSettings>;
  forgetDataset(path: string): Promise<AppSettings>;
  choose(options: OpenDialogOptions): Promise<string[]>;
  memory(): Promise<MemoryInfo>;
  /** Free space on the drive holding ``path`` (or its nearest existing parent). */
  freeSpace(path: string): Promise<number | null>;
  readTextFile(path: string): Promise<string>;
  /** Ask where to save, then write ``text`` there. Returns the path, or null if cancelled. */
  saveTextFile(options: { title: string; defaultName: string; text: string; extension: string }): Promise<string | null>;
  showInFolder(path: string): Promise<void>;
  /** The path of a file dropped on the page. */
  pathForFile(file: File): string;
  platform: string;
}
