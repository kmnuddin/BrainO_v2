/** Dataset overview: the dataset in one line, the running pipeline's progress, and a table of
 * subjects by pipeline step. */

import { useEffect, useMemo, useState } from "react";
import { ErrorNotice, JobLine, Loading, Pill, type Tone } from "../components/ui";
import type { PipelineState } from "../engine/types";
import { datasetName, duration, plural } from "../lib/format";
import { useApp } from "../state/app";
import { type SubjectStatus, useDataset } from "../state/dataset";
import { isActive } from "../state/jobs";

type Filter = "all" | "decision" | "running" | "todo";

export function Overview() {
  const { go } = useApp();
  const info = useDataset();
  const { summary } = info;
  const [chosen, setChosen] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>("all");

  // Show the running pipeline, else the one chosen, else the one with the most results.
  const pipeline = useMemo(() => {
    const named = (n: string | null) => info.pipelines.find((p) => p.item.name === n);
    const done = (s: PipelineState | null) =>
      s ? Object.values(s.subjects).reduce((n, steps) => n + Object.keys(steps).length, 0) : 0;
    const most = [...info.pipelines].sort((a, b) => done(b.state) - done(a.state))[0];
    return named(info.runningPipeline) ?? named(chosen) ?? most;
  }, [info.pipelines, info.runningPipeline, chosen]);

  useEffect(() => setFilter("all"), [pipeline?.item.name]);

  if (info.summaryError) return <main className="workspace"><ErrorNotice error={info.summaryError} /></main>;
  if (!summary) return <Loading what="Reading the dataset" />;

  const first = summary.recordings[0];
  const eeg = first?.channels.eeg;
  const facts = [
    plural(summary.n_subjects, "subject"),
    summary.sessions.length ? plural(summary.sessions.length, "session") : null,
    summary.tasks.length ? `task${summary.tasks.length > 1 ? "s" : ""} ${summary.tasks.join(", ")}` : null,
    eeg ? `${eeg} EEG channels` : null,
    first?.sampling_rate_hz ? `${first.sampling_rate_hz} Hz` : null,
    summary.total_duration_s ? `${duration(summary.total_duration_s)} recorded` : null,
  ].filter(Boolean);
  const pipelineJob = info.jobs.find((j) => j.kind === "pipeline" && isActive(j));
  const subjects = summary.subjects;
  const counts: Record<SubjectStatus, number> = { analysed: 0, decision: 0, running: 0, imported: 0 };
  for (const s of subjects) counts[info.subjectStatus(s)]++;
  const shown = subjects.filter((s) => {
    const status = info.subjectStatus(s);
    if (filter === "decision") return status === "decision";
    if (filter === "running") return status === "running";
    if (filter === "todo") return status === "imported";
    return true;
  });
  const steps = pipeline?.state?.steps ?? pipeline?.item.steps?.map((s) => s.id) ?? [];
  const recordingsOf = (s: string) => summary.recordings.filter((r) => r.subject === s);

  return (
    <main className="workspace">
      <div className="row" style={{ alignItems: "flex-end", gap: 16 }}>
        <div>
          <h1 style={{ marginBottom: 4 }}>{datasetName(summary.name, info.path)}</h1>
          <div className="muted">{facts.join(" · ")}</div>
        </div>
        <span className="spacer" />
        <button type="button" className="btn" onClick={() => go({ name: "import" })}>
          Import more
        </button>
        <button type="button" className="btn primary" onClick={() => go({ name: "pipelines" })}>
          Run a pipeline
        </button>
      </div>

      {subjects.length === 0 ? (
        <div className="card">
          <h2>No recordings yet</h2>
          <p className="muted">Import recordings to start.</p>
          <div>
            <button type="button" className="btn primary" onClick={() => go({ name: "import" })}>
              Import recordings
            </button>
          </div>
        </div>
      ) : (
        <>
          <section className="card compact">
            <div className="row">
              {info.pipelines.length > 1 && !pipelineJob ? (
                <label className="row small" style={{ gap: 8 }}>
                  <span className="muted">Pipeline</span>
                  <select
                    className="text"
                    value={pipeline?.item.name}
                    onChange={(e) => setChosen(e.target.value)}
                  >
                    {info.pipelines.map((p) => (
                      <option key={p.item.name} value={p.item.name}>
                        {p.item.name}
                      </option>
                    ))}
                  </select>
                </label>
              ) : (
                <span style={{ fontWeight: 600 }}>{pipeline ? pipeline.item.name : "No pipeline for this dataset's tasks"}</span>
              )}
              <span className="spacer" />
              <span className="small">
                <b>{counts.analysed}</b> analysed · <b>{counts.running}</b> running ·{" "}
                <b style={{ color: "var(--warn-text)" }}>{counts.decision}</b> waiting for your decision
              </span>
            </div>
            <div className="stacked-bar" aria-hidden="true">
              <div style={{ width: `${(100 * counts.analysed) / subjects.length}%`, background: "var(--ok)" }} />
              <div style={{ width: `${(100 * counts.running) / subjects.length}%`, background: "var(--accent)" }} />
              <div style={{ width: `${(100 * counts.decision) / subjects.length}%`, background: "var(--warn)" }} />
            </div>
            {pipelineJob ? <JobLine job={pipelineJob} /> : null}
          </section>

          <div className="chips" role="group" aria-label="Show">
            {(
              [
                ["all", `All ${subjects.length}`],
                ["decision", `Needs you ${counts.decision}`],
                ["running", `Running ${counts.running}`],
                ["todo", `Not analysed ${counts.imported}`],
              ] as [Filter, string][]
            ).map(([key, label]) => (
              <button key={key} type="button" className="chip" aria-pressed={filter === key} onClick={() => setFilter(key)}>
                {label}
              </button>
            ))}
          </div>

          <div className="table-wrap">
            <table className="table" style={{ minWidth: 720 }}>
              <thead>
                <tr>
                  <th>Subject</th>
                  <th>Recordings</th>
                  {steps.map((s) => (
                    <th key={s}>{s}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {shown.map((subject) => {
                  const recs = recordingsOf(subject);
                  return (
                    <tr key={subject}>
                      <td>
                        <button type="button" className="linkish mono" onClick={() => go({ name: "subject", subject })}>
                          sub-{subject}
                        </button>
                      </td>
                      <td className="muted">
                        {recs.length || ""}
                      </td>
                      {steps.map((step) => (
                        <td key={step}>
                          <StepCell
                            state={pipeline?.state ?? null}
                            subject={subject}
                            step={step}
                            running={info.runningPipeline === pipeline?.item.name}
                            onDecide={(proposal) => go({ name: "decisions", proposal })}
                          />
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {shown.length === 0 ? <div className="empty">No subjects match.</div> : null}
          </div>
          {summary.truncated ? (
            <p className="small muted">Only the first recordings' details are shown; the totals cover all.</p>
          ) : null}
        </>
      )}
    </main>
  );
}

function StepCell({
  state,
  subject,
  step,
  running,
  onDecide,
}: {
  state: PipelineState | null;
  subject: string;
  step: string;
  running: boolean;
  onDecide: (proposal: string) => void;
}) {
  const cell = state?.subjects[subject]?.[step];
  let tone: Tone = "todo";
  let text = running ? "Queued" : "Not run";
  if (cell?.status === "done" || cell?.status === "approved") {
    tone = "ok";
    text = "Done";
  } else if (cell?.status === "waiting") {
    if (cell.proposal_id) {
      return (
        <button type="button" className="pill warn" style={{ border: 0, cursor: "pointer" }} onClick={() => onDecide(cell.proposal_id!)}>
          Decide
        </button>
      );
    }
    tone = "warn";
    text = "Decide";
  } else if (cell?.status === "rejected") {
    tone = "danger";
    text = "Rejected";
  }
  return <Pill tone={tone}>{text}</Pill>;
}
