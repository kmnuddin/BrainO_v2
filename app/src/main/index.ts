/** The app's main process: one window, the engine it starts, and what the page may ask of
 * the computer. */

import { readFile, stat, statfs, writeFile } from "node:fs/promises";
import { freemem, totalmem } from "node:os";
import { dirname, extname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { BrowserWindow, app, dialog, ipcMain, shell } from "electron";
import type { AppSettings, HttpMethod, OpenDialogOptions } from "../shared/ipc";
import { Engine, engineCommand } from "./engine";
import { SettingsFile, defaultSettings, withOpened } from "./settings";

const here = dirname(fileURLToPath(import.meta.url));
const METHODS: HttpMethod[] = ["GET", "POST", "PUT", "DELETE"];
const MAX_TEXT_FILE = 5 * 2 ** 20;
const TEXT_FILES = [".csv", ".tsv", ".txt"];

const capturing = !!process.env.BRAINO_CAPTURE && !app.isPackaged;
let window: BrowserWindow | null = null;
let settings: SettingsFile;
let engine: Engine;

function send(channel: string, value: unknown): void {
  if (window && !window.isDestroyed()) window.webContents.send(channel, value);
}

function createWindow(): void {
  window = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 960,
    minHeight: 640,
    title: "BrainO",
    backgroundColor: "#EEF0F4",
    show: false,
    autoHideMenuBar: true,
    webPreferences: {
      preload: join(here, "../preload/index.cjs"),
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
      webSecurity: true,
      spellcheck: false,
    },
  });
  window.once("ready-to-show", () => window?.show());

  const capture = process.env.BRAINO_CAPTURE;
  if (capture && capturing) {
    window.webContents.once("did-finish-load", () => {
      void import("./capture")
        .then(({ runCapture }) => runCapture(window!, capture))
        .finally(() => app.quit());
    });
  }

  // The page never navigates away or opens windows; links to the web open in the browser.
  window.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith("https://")) void shell.openExternal(url);
    return { action: "deny" };
  });
  window.webContents.on("will-navigate", (event) => event.preventDefault());

  const devServer = process.env.ELECTRON_RENDERER_URL;
  if (!app.isPackaged && devServer) {
    void window.loadURL(devServer);
  } else {
    void window.loadFile(join(here, "../renderer/index.html"));
  }
}

function handle(): void {
  ipcMain.handle("engine:status", () => engine.getStatus());
  ipcMain.handle("engine:restart", () => engine.restart());
  ipcMain.handle("api:request", (_, method: unknown, path: unknown, body: unknown) => {
    if (!METHODS.includes(method as HttpMethod) || typeof path !== "string") {
      throw new Error("invalid request");
    }
    return engine.request(method as HttpMethod, path, body);
  });

  ipcMain.handle("settings:get", () => settings.get());
  ipcMain.handle("settings:update", async (_, changes: Partial<AppSettings>) => {
    const before = settings.get();
    const after = settings.update(changes ?? {});
    if (before.enginePython !== after.enginePython) await engine.restart();
    return after;
  });
  ipcMain.handle("settings:opened", (_, path: unknown) => {
    if (typeof path !== "string" || !path) throw new Error("invalid path");
    return settings.set(withOpened(settings.get(), path, new Date()));
  });
  ipcMain.handle("settings:forget", (_, path: unknown) => {
    const current = settings.get();
    return settings.set({ ...current, recent: current.recent.filter((r) => r.path !== path) });
  });

  ipcMain.handle("dialog:choose", async (_, options: OpenDialogOptions) => {
    if (!window) return [];
    if (capturing) {
      const { queuedChoices } = await import("./capture");
      return queuedChoices.shift() ?? [];
    }
    if (options.kind === "save") {
      const result = await dialog.showSaveDialog(window, {
        title: options.title,
        defaultPath: options.defaultPath,
        filters: options.filters,
      });
      return result.canceled || !result.filePath ? [] : [result.filePath];
    }
    const properties: ("openFile" | "openDirectory" | "multiSelections" | "createDirectory")[] = {
      file: ["openFile"] as const,
      files: ["openFile", "multiSelections"] as const,
      folder: ["openDirectory", "createDirectory"] as const,
      folders: ["openDirectory", "multiSelections"] as const,
    }[options.kind].slice();
    const result = await dialog.showOpenDialog(window, {
      title: options.title,
      defaultPath: options.defaultPath,
      filters: options.filters,
      properties,
    });
    return result.canceled ? [] : result.filePaths;
  });

  ipcMain.handle("system:memory", () => ({ totalBytes: totalmem(), freeBytes: freemem() }));
  ipcMain.handle("system:free-space", async (_, path: unknown) => {
    if (typeof path !== "string" || !path) return null;
    let folder = path;
    for (let i = 0; i < 32; i++) {
      try {
        const info = await statfs(folder);
        return info.bavail * info.bsize;
      } catch {
        const parent = dirname(folder);
        if (parent === folder) return null;
        folder = parent;
      }
    }
    return null;
  });
  // Only small text files (a mapping CSV) can be read, and only when the user chose them.
  ipcMain.handle("file:read-text", async (_, path: unknown) => {
    if (typeof path !== "string" || !TEXT_FILES.includes(extname(path).toLowerCase())) {
      throw new Error("only .csv, .tsv and .txt files can be read here");
    }
    if ((await stat(path)).size > MAX_TEXT_FILE) throw new Error("the file is larger than 5 MB");
    return readFile(path, "utf-8");
  });
  // Exports (a figure, a table): the user chooses where in a save dialog, so the page can only
  // write where the user said.
  ipcMain.handle(
    "file:save-text",
    async (_, options: { title: string; defaultName: string; text: string; extension: string }) => {
      if (!window || typeof options?.text !== "string" || !/^[a-z0-9]{1,8}$/.test(options.extension)) {
        throw new Error("invalid export");
      }
      const chosen = capturing
        ? (await import("./capture")).queuedChoices.shift()?.[0]
        : (
            await dialog.showSaveDialog(window, {
              title: options.title,
              defaultPath: join(app.getPath("documents"), options.defaultName),
              filters: [{ name: options.extension.toUpperCase(), extensions: [options.extension] }],
            })
          ).filePath;
      if (!chosen) return null;
      await writeFile(chosen, options.text, "utf-8");
      return chosen;
    },
  );
  ipcMain.handle("shell:show", (_, path: unknown) => {
    if (typeof path === "string" && path) shell.showItemInFolder(path);
  });
}

// A separate settings folder (for development and the capture script), so trying the app
// does not touch the real settings.
if (process.env.BRAINO_APP_DATA && !app.isPackaged) {
  app.setPath("userData", process.env.BRAINO_APP_DATA);
}

const single = app.requestSingleInstanceLock();
if (!single) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (window?.isMinimized()) window.restore();
    window?.focus();
  });

  app.whenReady().then(() => {
    settings = new SettingsFile(
      join(app.getPath("userData"), "settings.json"),
      defaultSettings(join(app.getPath("documents"), "BrainO")),
    );
    engine = new Engine(() =>
      engineCommand({
        configured: settings.get().enginePython,
        appPath: app.getAppPath(),
        resourcesPath: process.resourcesPath,
        packaged: app.isPackaged,
        env: process.env,
        platform: process.platform,
      }),
    );
    engine.on("status", (status) => send("engine:status", status));
    engine.on("event", (message) => send("engine:event", message));
    handle();
    engine.start();
    createWindow();
  });

  let quitting = false;
  app.on("before-quit", (event) => {
    if (quitting) return;
    event.preventDefault();
    quitting = true;
    void engine.stop().finally(() => app.quit());
  });
  app.on("window-all-closed", () => app.quit());
}
