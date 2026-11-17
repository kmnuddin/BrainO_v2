import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { engineCommand, parseReadyLine } from "./engine";
import { defaultSettings, parseSettings, withOpened } from "./settings";

describe("parseReadyLine", () => {
  it("reads the engine's ready line", () => {
    expect(parseReadyLine('{"braino_server": {"url": "http://127.0.0.1:5123", "pid": 7}}')).toEqual({
      url: "http://127.0.0.1:5123",
      pid: 7,
    });
  });

  it("ignores other lines and addresses off this computer", () => {
    expect(parseReadyLine("Loading…")).toBeNull();
    expect(parseReadyLine('{"other": 1}')).toBeNull();
    expect(parseReadyLine('{"braino_server": {"url": "http://10.0.0.5:80"}}')).toBeNull();
    expect(parseReadyLine("{broken")).toBeNull();
  });
});

describe("engineCommand", () => {
  const base = {
    configured: "",
    appPath: resolve("/repo/app"),
    resourcesPath: resolve("/opt/braino/resources"),
    packaged: false,
    env: {},
    platform: "win32" as NodeJS.Platform,
  };

  it("prefers BRAINO_PYTHON, then the setting", () => {
    expect(engineCommand({ ...base, env: { BRAINO_PYTHON: "C:/py/python.exe" }, configured: "x" })).toEqual([
      "C:/py/python.exe",
      "-m",
      "braino",
    ]);
    expect(engineCommand({ ...base, configured: "/usr/bin/python3" })).toEqual(["/usr/bin/python3", "-m", "braino"]);
  });

  it("uses the engine's virtual environment when developing", () => {
    const venv = join(resolve("/repo/engine/.venv"), "Scripts", "python.exe");
    expect(engineCommand({ ...base, exists: (p) => p === venv })).toEqual([venv, "-m", "braino"]);
  });

  it("uses the bundled Python in an installed app, else the braino command", () => {
    const bundled = join(resolve("/opt/braino/resources"), "engine", "bin", "python");
    expect(
      engineCommand({ ...base, packaged: true, platform: "linux", exists: (p) => p === bundled }),
    ).toEqual([bundled, "-m", "braino"]);
    expect(engineCommand({ ...base, exists: () => false })).toEqual(["braino"]);
  });
});

describe("settings", () => {
  const defaults = defaultSettings("/home/me/Documents/BrainO");

  it("keeps known fields of the right type", () => {
    const parsed = parseSettings(
      JSON.stringify({ firstRunDone: true, datasetsFolder: 5, recent: [{ path: "/d", openedAt: "x" }, { path: 1 }] }),
      defaults,
    );
    expect(parsed.firstRunDone).toBe(true);
    expect(parsed.datasetsFolder).toBe(defaults.datasetsFolder);
    expect(parsed.recent).toEqual([{ path: "/d", openedAt: "x" }]);
    expect(parseSettings("not json", defaults)).toEqual(defaults);
  });

  it("puts an opened dataset first, once", () => {
    const now = new Date("2026-11-17T10:00:00Z");
    const one = withOpened(defaults, "C:\\data\\a", now);
    const two = withOpened(withOpened(one, "C:\\data\\b", now), "c:\\DATA\\A", now);
    expect(two.recent.map((r) => r.path)).toEqual(["c:\\DATA\\A", "C:\\data\\b"]);
  });
});
