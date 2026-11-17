/** Settings: folders, the engine's Python, and the engine's state. */

import { useEffect, useState } from "react";
import { useApp } from "../state/app";

export function Settings() {
  const { settings, updateSettings, engine, system } = useApp();
  const [folder, setFolder] = useState("");
  const [python, setPython] = useState("");
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (settings) {
      setFolder(settings.datasetsFolder);
      setPython(settings.enginePython);
    }
  }, [settings]);

  const save = async () => {
    await updateSettings({ datasetsFolder: folder, enginePython: python.trim() });
    setSaved(true);
    window.setTimeout(() => setSaved(false), 2000);
  };

  return (
    <main className="workspace narrow">
      <h1>Settings</h1>
      <section className="card">
        <h2>Folders</h2>
        <label className="field">
          Folder for new datasets
          <div className="row" style={{ gap: 8, flexWrap: "nowrap" }}>
            <input className="text mono" style={{ flex: 1 }} value={folder} onChange={(e) => setFolder(e.target.value)} />
            <button
              type="button"
              className="btn"
              onClick={() =>
                void window.braino.choose({ kind: "folder", defaultPath: folder }).then(([p]) => p && setFolder(p))
              }
            >
              Choose…
            </button>
          </div>
        </label>
        {system ? (
          <p className="small muted">
            Your BrainO folder (saved pipelines, subject keys): <span className="mono">{system.user_folder}</span>
          </p>
        ) : null}
      </section>

      <section className="card">
        <h2>Engine</h2>
        <label className="field">
          Python with BrainO installed (empty: find it)
          <div className="row" style={{ gap: 8, flexWrap: "nowrap" }}>
            <input className="text mono" style={{ flex: 1 }} value={python} onChange={(e) => setPython(e.target.value)} />
            <button
              type="button"
              className="btn"
              onClick={() => void window.braino.choose({ kind: "file", title: "Python" }).then(([p]) => p && setPython(p))}
            >
              Choose…
            </button>
          </div>
        </label>
        <p className="small muted">
          {engine.state === "running" && system ? `BrainO ${system.version} is running` : `The engine is ${engine.state}`}
          {engine.command ? (
            <>
              {" "}
              with <span className="mono">{engine.command.join(" ")}</span>
            </>
          ) : null}
          .{system ? ` Up to ${system.jobs.max_parallel} jobs run at once on this computer.` : ""}
        </p>
        {engine.error ? <div className="notice error">{engine.error}</div> : null}
        <div className="row">
          <button type="button" className="btn" onClick={() => void window.braino.restartEngine()}>
            Restart the engine
          </button>
        </div>
      </section>

      <div className="row">
        <span className="spacer" />
        {saved ? <span className="small" style={{ color: "var(--ok)" }}>Saved</span> : null}
        <button type="button" className="btn primary big" onClick={() => void save()}>
          Save
        </button>
      </div>
    </main>
  );
}
