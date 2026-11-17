/** The only bridge between the page and the computer. The page gets no Node.js and never
 * sees the engine's token: requests go through the main process. */

import { contextBridge, ipcRenderer, webUtils } from "electron";
import type { BrainoBridge, EngineStatus } from "../shared/ipc";

function listen<T>(channel: string, listener: (value: T) => void): () => void {
  const handler = (_: unknown, value: T) => listener(value);
  ipcRenderer.on(channel, handler);
  return () => {
    ipcRenderer.removeListener(channel, handler);
  };
}

const bridge: BrainoBridge = {
  engineStatus: () => ipcRenderer.invoke("engine:status"),
  restartEngine: () => ipcRenderer.invoke("engine:restart"),
  onEngineStatus: (listener) => listen<EngineStatus>("engine:status", listener),
  request: (method, path, body) => ipcRenderer.invoke("api:request", method, path, body),
  onEvent: (listener) => listen<unknown>("engine:event", listener),
  settings: () => ipcRenderer.invoke("settings:get"),
  updateSettings: (changes) => ipcRenderer.invoke("settings:update", changes),
  openedDataset: (path) => ipcRenderer.invoke("settings:opened", path),
  forgetDataset: (path) => ipcRenderer.invoke("settings:forget", path),
  choose: (options) => ipcRenderer.invoke("dialog:choose", options),
  memory: () => ipcRenderer.invoke("system:memory"),
  freeSpace: (path) => ipcRenderer.invoke("system:free-space", path),
  readTextFile: (path) => ipcRenderer.invoke("file:read-text", path),
  saveTextFile: (options) => ipcRenderer.invoke("file:save-text", options),
  showInFolder: (path) => ipcRenderer.invoke("shell:show", path),
  pathForFile: (file) => webUtils.getPathForFile(file),
  platform: process.platform,
};

contextBridge.exposeInMainWorld("braino", bridge);
