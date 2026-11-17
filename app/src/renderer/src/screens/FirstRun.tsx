/** First run: the computer, the AI model download, and where new datasets go. */

import { useEffect, useState } from "react";
import * as Icon from "../components/icons";
import { ErrorNotice, JobLine, Loading } from "../components/ui";
import { api } from "../engine/api";
import type { Job } from "../engine/types";
import { gigabytes } from "../lib/format";
import { useApp, useJobs } from "../state/app";
import { isActive } from "../state/jobs";

const NEEDED_GB = 3;

export function FirstRun() {
  const { system, settings, updateSettings, go } = useApp();
  const jobs = useJobs();
  const [folder, setFolder] = useState(settings?.datasetsFolder ?? "");
  const [free, setFree] = useState<number | null>(null);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    if (settings && !folder) setFolder(settings.datasetsFolder);
  }, [settings, folder]);
  useEffect(() => {
    if (folder) void window.braino.freeSpace(folder).then(setFree);
  }, [folder]);

  if (!system) return <Loading what="Checking this computer" />;

  const hw = system.hardware;
  const rec = system.recommendation;
  const model = system.models.find((m) => m.id === rec.model);
  const installed = system.installed_models.includes(rec.model);
  const setupJob: Job | undefined = jobs.find((j) => j.kind === "model_setup");
  const downloading = setupJob && isActive(setupJob);
  const gpu = hw.gpus.find((g) => g.name === rec.gpu) ?? hw.gpus[0];
  const enoughMemory = (hw.memory_gb ?? 0) >= 15;

  const download = async () => {
    setError(null);
    try {
      await api.modelSetup(rec.model, rec.backend);
    } catch (e) {
      setError(e);
    }
  };
  const choose = async () => {
    const [chosen] = await window.braino.choose({
      kind: "folder",
      title: "Folder for new datasets",
      defaultPath: folder,
    });
    if (chosen) setFolder(chosen);
  };
  const finish = async () => {
    await updateSettings({ datasetsFolder: folder, firstRunDone: true });
    go({ name: "home" });
  };

  return (
    <main className="workspace narrow" style={{ paddingTop: 40 }}>
      <div>
        <h1 style={{ fontSize: 28, marginBottom: 6 }}>Set up BrainO</h1>
        <p className="muted" style={{ fontSize: 15 }}>
          Three steps, once. Everything runs on this computer, and nothing you load leaves it.
        </p>
      </div>

      <section className="card">
        <div className="row">
          <span className="step-circle done">
            <Icon.Check />
          </span>
          <h2>Your computer</h2>
          <span className="spacer" />
          <span className="small" style={{ color: "var(--ok)", fontWeight: 600 }}>
            Checked
          </span>
        </div>
        <div className="grid cols-4">
          <div className="tile">
            <div className="label">Processor</div>
            <div className="value">{hw.cpu ?? "Unknown"}</div>
            <div className="tiny muted">{hw.cpu_count ? `${hw.cpu_count} logical processors` : ""}</div>
          </div>
          <div className="tile">
            <div className="label">Memory</div>
            <div className="value">{hw.memory_gb ? `${Math.round(hw.memory_gb)} GB` : "Unknown"}</div>
            <div className="tiny" style={{ color: enoughMemory ? "var(--muted)" : "var(--warn-text)" }}>
              {enoughMemory ? "Enough for the agent" : "Below the 16 GB BrainO is designed for"}
            </div>
          </div>
          <div className="tile">
            <div className="label">Graphics card</div>
            <div className="value">{gpu ? gpu.name : "None found"}</div>
            <div className="tiny muted">
              {rec.placement === "cpu" ? "The agent uses the processor" : "The agent uses the graphics card"}
            </div>
          </div>
          <div className="tile">
            <div className="label">Free space</div>
            <div className="value">{free == null ? "Unknown" : gigabytes(free, 0)}</div>
            <div className="tiny muted">{NEEDED_GB} GB needed for the model</div>
          </div>
        </div>
        {rec.warnings.map((w) => (
          <div key={w} className="notice warn">
            {w}
          </div>
        ))}
      </section>

      <section className={`card${installed ? "" : " selected"}`}>
        <div className="row">
          <span className={`step-circle ${installed ? "done" : "active"}`}>
            {installed ? <Icon.Check /> : "2"}
          </span>
          <h2>The AI model</h2>
          <span className="spacer" />
          <span className="small" style={{ fontWeight: 600, color: installed ? "var(--ok)" : "var(--accent-text)" }}>
            {installed ? "Ready" : downloading ? "Downloading" : "Not downloaded"}
          </span>
        </div>
        <div className="row" style={{ alignItems: "baseline", gap: 8 }}>
          <span style={{ fontWeight: 600 }}>{model?.name ?? rec.model}</span>
          <span className="small muted">
            recommended for this computer{model ? ` · ${model.size_gb.toFixed(1)} GB` : ""} · runs on the{" "}
            {rec.placement === "cpu" ? "processor" : "graphics card"}
          </span>
        </div>
        {setupJob && (downloading || setupJob.status === "failed") ? <JobLine job={setupJob} /> : null}
        <p className="small muted">
          Downloaded once with the documentation search model, and checked against its published
          fingerprint. Other models can be chosen later with <span className="mono">braino model setup --model</span>.
        </p>
        <ErrorNotice error={error} />
        {!installed && !downloading ? (
          <div>
            <button type="button" className="btn primary" onClick={() => void download()}>
              Download the model
            </button>
          </div>
        ) : null}
      </section>

      <section className="card">
        <div className="row">
          <span className="step-circle">3</span>
          <h2>Where your data lives</h2>
        </div>
        <label className="field" htmlFor="datadir">
          Folder for new datasets
        </label>
        <div className="row" style={{ gap: 8, flexWrap: "nowrap" }}>
          <input
            id="datadir"
            className="text mono"
            style={{ flex: 1 }}
            value={folder}
            onChange={(e) => setFolder(e.target.value)}
          />
          <button type="button" className="btn" onClick={() => void choose()}>
            Choose…
          </button>
        </div>
        <p className="small muted">
          Datasets are kept as BIDS folders. You can also open a dataset from anywhere.
        </p>
      </section>

      <div className="row">
        <span className="small muted">
          {downloading ? "You can start working while the model downloads." : ""}
        </span>
        <span className="spacer" />
        {!installed ? (
          <button type="button" className="btn link big" disabled={!folder} onClick={() => void finish()}>
            Use BrainO without the agent
          </button>
        ) : null}
        <button type="button" className="btn primary big" disabled={!folder} onClick={() => void finish()}>
          Continue
        </button>
      </div>
    </main>
  );
}
