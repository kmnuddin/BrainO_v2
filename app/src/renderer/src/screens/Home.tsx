/** Home: open a dataset, start one from recordings, try the example, and recent datasets. */

import { useState } from "react";
import * as Icon from "../components/icons";
import { ErrorNotice, JobLine } from "../components/ui";
import { ApiError, api } from "../engine/api";
import type { RecentDataset } from "../../../shared/ipc";
import { datasetName, plural, whenOpened } from "../lib/format";
import { useAsync } from "../lib/useAsync";
import { useApp, useJobs } from "../state/app";
import { isActive } from "../state/jobs";

export function Home() {
  const { settings, openDataset, go, engine, jobs: store } = useApp();
  const jobs = useJobs();
  const [error, setError] = useState<unknown>(null);
  const [exampleOpen, setExampleOpen] = useState(false);
  const exampleJob = jobs.find((j) => j.kind === "example");
  const ready = engine.state === "running";

  const open = async () => {
    const [folder] = await window.braino.choose({ kind: "folder", title: "Open a BIDS dataset" });
    if (!folder) return;
    setError(null);
    try {
      const summary = await api.datasetSummary(folder, 0);
      if (summary.n_recordings === 0 && !summary.bids_version) {
        setError(
          `${folder} is not a BIDS dataset: it has no dataset_description.json and no recordings. ` +
            "To make a dataset from recordings, choose New dataset from recordings.",
        );
        return;
      }
      await openDataset(folder);
    } catch (e) {
      setError(e);
    }
  };

  // Opens the example when its download finishes.
  const example = async (subjects: number) => {
    setError(null);
    setExampleOpen(false);
    try {
      const job = await store.finished((await api.example("P3", subjects)).id);
      if (job.status !== "done") return;
      const { result } = await api.job(job.id);
      const dataset = (result as { dataset?: string } | null)?.dataset;
      if (dataset) await openDataset(dataset);
    } catch (e) {
      setError(e);
    }
  };

  return (
    <main className="workspace" style={{ padding: 32, gap: 28 }}>
      <h1 style={{ fontSize: 26 }}>Datasets</h1>
      <div className="grid cols-3">
        <button type="button" className="action-card" disabled={!ready} onClick={() => void open()}>
          <Icon.Folder />
          <span className="title">Open a dataset</span>
          <span className="small muted">A BIDS folder already on this computer.</span>
        </button>
        <button
          type="button"
          className="action-card"
          disabled={!ready}
          onClick={() => go({ name: "import", newDataset: true })}
        >
          <Icon.Download />
          <span className="title">New dataset from recordings</span>
          <span className="small muted">
            EDF, BDF, BrainVision, EEGLAB, FIF and more. Personal details are removed.
          </span>
        </button>
        <button
          type="button"
          className="action-card"
          disabled={!ready || (exampleJob && isActive(exampleJob))}
          onClick={() => setExampleOpen((o) => !o)}
        >
          <Icon.Wave />
          <span className="title">Try an example</span>
          <span className="small muted">
            ERP CORE P3: download it and reproduce the published effect.
          </span>
        </button>
      </div>

      {exampleOpen ? (
        <section className="card compact">
          <h2>ERP CORE P3</h2>
          <p className="small muted">
            Public data from Kappenman et al. (2021), CC-BY-4.0: young adults seeing rare and
            frequent letters. Downloaded from OSF and checked against pinned fingerprints.
          </p>
          <div className="row">
            <button type="button" className="btn primary" onClick={() => void example(10)}>
              10 subjects · 0.5 GB
            </button>
            <button type="button" className="btn" onClick={() => void example(40)}>
              All 40 subjects · 2.1 GB
            </button>
            <span className="small muted">All 40 are needed to compare with the published scores.</span>
          </div>
        </section>
      ) : null}
      {exampleJob && (isActive(exampleJob) || exampleJob.status === "failed") ? (
        <section className="card compact">
          <JobLine job={exampleJob} />
        </section>
      ) : null}
      <ErrorNotice error={error} />

      <section className="stack" style={{ gap: 10 }}>
        <h2>Recent</h2>
        {settings && settings.recent.length > 0 ? (
          <div className="table-wrap">
            <table className="table" style={{ minWidth: 640 }}>
              <thead>
                <tr>
                  <th>Dataset</th>
                  <th>Subjects</th>
                  <th>Tasks</th>
                  <th>Waiting for you</th>
                  <th>Opened</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {settings.recent.map((r) => (
                  <RecentRow key={r.path} recent={r} ready={ready} />
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="card compact muted">Datasets you open will be listed here.</div>
        )}
      </section>
    </main>
  );
}

function RecentRow({ recent, ready }: { recent: RecentDataset; ready: boolean }) {
  const { openDataset, setSettings } = useApp();
  const info = useAsync(async () => {
    if (!ready) return null;
    const [summary, pending] = await Promise.all([
      api.datasetSummary(recent.path, 0),
      api.proposals(recent.path, "pending"),
    ]);
    return { summary, pending: pending.length };
  }, [recent.path, ready]);
  const missing = info.error instanceof ApiError && info.error.status === 400;

  return (
    <tr>
      <td>
        <button
          type="button"
          className="linkish"
          style={{ fontWeight: 600 }}
          disabled={missing || !ready}
          onClick={() => void openDataset(recent.path)}
        >
          {datasetName(info.data?.summary.name, recent.path)}
        </button>
        <div className="tiny muted mono" style={{ wordBreak: "break-all" }}>
          {recent.path}
        </div>
      </td>
      <td>{missing ? "" : (info.data?.summary.n_subjects ?? "")}</td>
      <td>{info.data?.summary.tasks.join(", ")}</td>
      <td>
        {missing ? (
          <span className="pill danger">Folder not found</span>
        ) : info.data && info.data.pending > 0 ? (
          <span className="pill warn">{plural(info.data.pending, "decision")}</span>
        ) : info.data ? (
          <span className="muted">Nothing</span>
        ) : null}
      </td>
      <td className="muted">{whenOpened(recent.openedAt)}</td>
      <td>
        <button
          type="button"
          className="btn small link"
          onClick={() => void window.braino.forgetDataset(recent.path).then(setSettings)}
        >
          Remove from list
        </button>
      </td>
    </tr>
  );
}
