/** The app's own settings, in a JSON file in Electron's user-data folder. */

import { mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import type { AppSettings } from "../shared/ipc";

const MAX_RECENT = 12;

export function defaultSettings(documents: string): AppSettings {
  return {
    datasetsFolder: documents,
    firstRunDone: false,
    recent: [],
    enginePython: "",
    agentPanelOpen: true,
  };
}

/** Settings read from ``text``, with anything missing or of the wrong type left at its
 * default. */
export function parseSettings(text: string, defaults: AppSettings): AppSettings {
  let raw: Record<string, unknown> = {};
  try {
    const value: unknown = JSON.parse(text);
    if (value && typeof value === "object") raw = value as Record<string, unknown>;
  } catch {
    return defaults;
  }
  const settings = { ...defaults };
  if (typeof raw.datasetsFolder === "string" && raw.datasetsFolder) {
    settings.datasetsFolder = raw.datasetsFolder;
  }
  if (typeof raw.firstRunDone === "boolean") settings.firstRunDone = raw.firstRunDone;
  if (typeof raw.enginePython === "string") settings.enginePython = raw.enginePython;
  if (typeof raw.agentPanelOpen === "boolean") settings.agentPanelOpen = raw.agentPanelOpen;
  if (Array.isArray(raw.recent)) {
    settings.recent = raw.recent
      .filter(
        (r): r is { path: string; openedAt: string } =>
          !!r && typeof r.path === "string" && typeof r.openedAt === "string",
      )
      .slice(0, MAX_RECENT);
  }
  return settings;
}

/** ``path`` first among the recent datasets, opened now. */
export function withOpened(settings: AppSettings, path: string, now: Date): AppSettings {
  const same = (a: string) => a.toLowerCase() === path.toLowerCase();
  const recent = [
    { path, openedAt: now.toISOString() },
    ...settings.recent.filter((r) => !same(r.path)),
  ].slice(0, MAX_RECENT);
  return { ...settings, recent };
}

export class SettingsFile {
  private value: AppSettings;

  constructor(
    private readonly path: string,
    defaults: AppSettings,
  ) {
    let text = "";
    try {
      text = readFileSync(path, "utf-8");
    } catch {
      // first start
    }
    this.value = text ? parseSettings(text, defaults) : defaults;
  }

  get(): AppSettings {
    return structuredClone(this.value);
  }

  set(value: AppSettings): AppSettings {
    this.value = value;
    mkdirSync(dirname(this.path), { recursive: true });
    const tmp = `${this.path}.tmp`;
    writeFileSync(tmp, JSON.stringify(value, null, 2), "utf-8");
    renameSync(tmp, this.path);
    return this.get();
  }

  update(changes: Partial<AppSettings>): AppSettings {
    const allowed: (keyof AppSettings)[] = [
      "datasetsFolder",
      "firstRunDone",
      "enginePython",
      "agentPanelOpen",
    ];
    const next = { ...this.value };
    for (const key of allowed) {
      if (key in changes) (next as Record<string, unknown>)[key] = changes[key];
    }
    return this.set(parseSettings(JSON.stringify(next), this.value));
  }
}
