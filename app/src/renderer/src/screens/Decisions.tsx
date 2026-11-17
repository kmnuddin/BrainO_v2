/** Decisions: one inbox for every proposal (from pipelines and the agent), each with its
 * evidence, to approve, edit or reject; the decided ones below. Decisions are recorded and
 * replayed when the pipeline runs again. */

import { useEffect, useMemo, useState } from "react";
import { HeadMap, Legend, LineChart } from "../components/charts";
import { ErrorNotice, JobLine, Loading, Pill } from "../components/ui";
import { api } from "../engine/api";
import type { Evidence, Job, Proposal } from "../engine/types";
import { plural } from "../lib/format";
import { useAsync } from "../lib/useAsync";
import { useApp, useJobs } from "../state/app";
import { useDataset } from "../state/dataset";
import { isActive } from "../state/jobs";

const TOOL_NAMES: Record<string, string> = {
  "eeg.detect_bad_channels": "Bad channels",
  "eeg.ica_clean": "ICA components",
  "analysis.erp_comparison": "ERP comparison plan",
  "analysis.spectral_comparison": "Spectral comparison plan",
};

export function Decisions({ initial }: { initial?: string }) {
  const { datasetVersion } = useApp();
  const info = useDataset();
  const all = useAsync(() => api.proposals(info.path), [info.path, datasetVersion]);
  const [selected, setSelected] = useState<string | null>(initial ?? null);
  const [showDecided, setShowDecided] = useState(false);

  const pending = useMemo(() => (all.data ?? []).filter((p) => p.status === "pending"), [all.data]);
  const decided = useMemo(
    () =>
      (all.data ?? [])
        .filter((p) => p.status !== "pending")
        .sort((a, b) => (b.decision?.decided_at ?? "").localeCompare(a.decision?.decided_at ?? "")),
    [all.data],
  );

  useEffect(() => {
    if (!all.data) return;
    const exists = all.data.some((p) => p.proposal_id === selected);
    if (!exists) setSelected(pending[0]?.proposal_id ?? null);
  }, [all.data, pending, selected]);

  if (all.error) return <main className="workspace"><ErrorNotice error={all.error} /></main>;
  if (!all.data) return <Loading what="Reading the proposals" />;
  const current = all.data.find((p) => p.proposal_id === selected) ?? null;
  const list = showDecided ? decided : pending;

  return (
    <main className="workspace">
      <div className="row">
        <h1>Decisions</h1>
        <span className="muted">
          {pending.length ? `${plural(pending.length, "proposal")} waiting for you` : "Nothing waiting for you"}
        </span>
      </div>
      <div className="split">
        <nav className="card compact" aria-label="Proposals" style={{ gap: 4 }}>
          <div className="chips" role="group" aria-label="Show" style={{ padding: "0 4px 8px" }}>
            <button type="button" className="chip" aria-pressed={!showDecided} onClick={() => setShowDecided(false)}>
              Waiting {pending.length}
            </button>
            <button type="button" className="chip" aria-pressed={showDecided} onClick={() => setShowDecided(true)}>
              Decided {decided.length}
            </button>
          </div>
          {list.length === 0 ? <div className="small muted" style={{ padding: 12 }}>None.</div> : null}
          {list.map((p) => (
            <button
              key={p.proposal_id}
              type="button"
              className="list-item"
              aria-current={p.proposal_id === selected}
              onClick={() => setSelected(p.proposal_id)}
            >
              <span className="row" style={{ gap: 6 }}>
                <span className="mono" style={{ fontWeight: 600 }}>
                  {subjectOf(p)}
                </span>
                <span>{TOOL_NAMES[p.tool] ?? p.tool}</span>
                {p.status !== "pending" ? (
                  <Pill tone={p.status === "approved" ? "ok" : "danger"}>{p.status === "approved" ? "Approved" : "Rejected"}</Pill>
                ) : null}
              </span>
              <span className="tiny muted">{summaryOf(p)}</span>
            </button>
          ))}
        </nav>
        <div className="stack" style={{ minWidth: 0 }}>
          {current ? (
            <ProposalView key={current.proposal_id} proposal={current} onDecided={() => { all.reload(); info.reload(); }} />
          ) : (
            <div className="card muted">
              {pending.length === 0
                ? "Nothing to decide. Proposals appear here when a pipeline, a tool or the agent asks you."
                : "Choose a proposal."}
            </div>
          )}
        </div>
      </div>
    </main>
  );
}

function subjectOf(p: Proposal): string {
  const subject = p.arguments.subject;
  return typeof subject === "string" ? `sub-${subject}` : "Dataset";
}

function summaryOf(p: Proposal): string {
  const proposal = p.proposal;
  if (p.tool === "eeg.detect_bad_channels") {
    const channels = (proposal.channels as string[]) ?? [];
    return channels.length ? `Proposed: ${channels.join(", ")}` : "No bad channels found";
  }
  if (p.tool === "eeg.ica_clean") {
    const exclude = (proposal.exclude as number[]) ?? [];
    return `${plural(exclude.length, "component")} to remove of ${proposal.n_components}`;
  }
  return p.tool;
}

function ProposalView({ proposal, onDecided }: { proposal: Proposal; onDecided: () => void }) {
  const info = useDataset();
  const jobs = useJobs();
  const { jobs: store } = useApp();
  const evidence = useAsync(() => api.evidence(info.path, proposal.proposal_id), [proposal.proposal_id]);
  const [note, setNote] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const job: Job | undefined = jobs.find((j) => j.id === jobId);
  const pending = proposal.status === "pending";
  const p = proposal.proposal;

  // The edit: the channels or components the person keeps in the proposal.
  const initial = proposal.tool === "eeg.detect_bad_channels" ? ((p.channels as string[]) ?? []) : ((p.exclude as number[]) ?? []).map(String);
  const [chosen, setChosen] = useState<Set<string>>(new Set(initial));
  const edited = chosen.size !== initial.length || initial.some((v) => !chosen.has(v));
  const toggle = (v: string) => {
    const next = new Set(chosen);
    if (next.has(v)) next.delete(v);
    else next.add(v);
    setChosen(next);
  };

  // The pipeline whose step waits for this proposal, to continue it after the decision.
  const pipeline = info.pipelines.find(({ state }) =>
    Object.values(state?.subjects ?? {}).some((steps) => Object.values(steps).some((s) => s.proposal_id === proposal.proposal_id)),
  );

  const decide = async (approve: boolean) => {
    setError(null);
    let edits: Record<string, unknown> | null = null;
    if (approve && edited) {
      edits =
        proposal.tool === "eeg.detect_bad_channels"
          ? { channels: [...chosen] }
          : { exclude: [...chosen].map(Number).sort((a, b) => a - b) };
    }
    try {
      const started = await api.decide(info.path, proposal.proposal_id, approve, edits, note || null);
      setJobId(started.id);
      const finished = await store.finished(started.id);
      if (finished.status === "done") onDecided();
    } catch (e) {
      setError(e);
    }
  };

  const continuePipeline = async () => {
    if (!pipeline) return;
    try {
      const started = await api.runPipeline({ dataset: info.path, name: pipeline.item.name });
      setJobId(started.id);
    } catch (e) {
      setError(e);
    }
  };

  return (
    <>
      <section className="card">
        <div className="row">
          <h2>
            <span className="mono">{subjectOf(proposal)}</span> · {TOOL_NAMES[proposal.tool] ?? proposal.tool}
          </h2>
          {pipeline ? <span className="small muted">from the pipeline {pipeline.item.name}</span> : null}
          <span className="spacer" />
          {pending ? <Pill tone="warn">Waiting for you</Pill> : <Pill tone={proposal.status === "approved" ? "ok" : "danger"}>{proposal.status}</Pill>}
        </div>
        {((p.notes as string[]) ?? []).map((n) => (
          <div key={n} className="small muted">
            {n}
          </div>
        ))}
        {evidence.error ? <ErrorNotice error={evidence.error} title="The evidence could not be read" /> : null}
        {!evidence.data && !evidence.error ? <Loading what="Reading the recording" /> : null}
        {evidence.data ? (
          <EvidenceView
            evidence={evidence.data}
            proposal={proposal}
            chosen={chosen}
            onToggle={pending ? toggle : undefined}
          />
        ) : null}
      </section>

      {pending ? (
        <section className="card">
          {edited ? (
            <div className="notice warn">
              You changed the proposal: approving applies{" "}
              {proposal.tool === "eeg.detect_bad_channels"
                ? chosen.size
                  ? `bad channels ${[...chosen].join(", ")}`
                  : "no bad channels"
                : `removing ${plural(chosen.size, "component")}`}
              . The change is recorded with your decision.
            </div>
          ) : null}
          <label className="field">
            Note (optional, kept with the decision)
            <input className="text" value={note} onChange={(e) => setNote(e.target.value)} placeholder="Why you decided so" />
          </label>
          <ErrorNotice error={error} />
          {job ? <JobLine job={job} cancellable={false} /> : null}
          <div className="row">
            <button type="button" className="btn danger" disabled={!!job && isActive(job)} onClick={() => void decide(false)}>
              Reject
            </button>
            <span className="spacer" />
            <button type="button" className="btn primary" disabled={!!job && isActive(job)} onClick={() => void decide(true)}>
              {edited ? "Approve as changed" : "Approve"}
            </button>
          </div>
        </section>
      ) : (
        <section className="card compact">
          <div className="small">
            {proposal.status === "approved" ? "Approved" : "Rejected"} by {proposal.decision?.decided_by}
            {proposal.decision ? ` on ${new Date(proposal.decision.decided_at).toLocaleString()}` : ""}
            {proposal.decision?.edited ? " (changed before approving)" : ""}.
          </div>
          {proposal.decision?.note ? <div className="small muted">“{proposal.decision.note}”</div> : null}
          {pipeline && proposal.status === "approved" ? (
            <div className="row">
              <span className="small muted">The pipeline continues from here when it runs again.</span>
              <span className="spacer" />
              <button type="button" className="btn" onClick={() => void continuePipeline()}>
                Run {pipeline.item.name} again
              </button>
            </div>
          ) : null}
          <ErrorNotice error={error} />
          {job ? <JobLine job={job} /> : null}
        </section>
      )}
    </>
  );
}

const TRACE = "var(--warn)";
const NEIGHBOUR = "var(--ok)";

function EvidenceView({
  evidence,
  proposal,
  chosen,
  onToggle,
}: {
  evidence: Evidence;
  proposal: Proposal;
  chosen: Set<string>;
  onToggle?: (v: string) => void;
}) {
  if (evidence.kind === "bad_channels") {
    const names = evidence.layout?.names ?? Object.keys(evidence.amplitude_uv);
    const neighbours = evidence.channels.map((c) => c.neighbour).filter((n): n is string => !!n);
    const x = evidence.channels[0]?.signal.map((_, i) => evidence.snippet.start_s + i / evidence.snippet.sfreq) ?? [];
    const median = medianOf(Object.values(evidence.amplitude_uv));
    return (
      <div className="stack">
        <div className="row" style={{ alignItems: "flex-start", gap: 24 }}>
          {evidence.layout ? (
            <HeadMap layout={evidence.layout} marked={[...chosen]} secondary={neighbours} title="Orange: marked bad · green: neighbours shown" />
          ) : null}
          <div className="stack" style={{ flex: 1, minWidth: 240 }}>
            <div className="small muted">
              {onToggle ? "Tick the channels to mark bad. You can add a channel the screen did not find." : "Channels marked bad"}
            </div>
            <div className="chips">
              {names.map((n) => {
                const amp = evidence.amplitude_uv[n];
                const proposed = evidence.proposed.includes(n);
                if (!onToggle && !chosen.has(n)) return null;
                return (
                  <button
                    key={n}
                    type="button"
                    className="chip mono"
                    aria-pressed={chosen.has(n)}
                    disabled={!onToggle}
                    title={amp != null ? `${amp} µV (median ${median.toFixed(1)} µV)` : undefined}
                    style={proposed ? undefined : { opacity: chosen.has(n) ? 1 : 0.7 }}
                    onClick={() => onToggle?.(n)}
                  >
                    {n}
                  </button>
                );
              })}
            </div>
            {proposal.status === "pending" && evidence.proposed.length === 0 ? (
              <div className="notice ok">The screen found no bad channels. Approve to record that, or tick any you see.</div>
            ) : null}
          </div>
        </div>
        {evidence.channels.map((c) => (
          <div key={c.name} className="stack" style={{ gap: 4 }}>
            <div className="row small" style={{ gap: 8 }}>
              <b className="mono">{c.name}</b>
              <span className="muted">{c.reasons.join("; ")}</span>
            </div>
            <LineChart
              x={x}
              series={[
                { label: c.name, values: c.signal, color: TRACE },
                ...(c.neighbour ? [{ label: `${c.neighbour} (good neighbour)`, values: c.neighbour_signal, color: NEIGHBOUR }] : []),
              ]}
              xLabel="time in the recording (s)"
              yLabel="µV"
              height={160}
            />
            <Legend
              series={[
                { label: c.name, color: TRACE },
                ...(c.neighbour ? [{ label: `${c.neighbour} (nearest good channel)`, color: NEIGHBOUR }] : []),
              ]}
            />
          </div>
        ))}
      </div>
    );
  }
  if (evidence.kind === "ica") {
    const layout = evidence.layout;
    return (
      <div className="stack">
        <div className="small muted">
          {onToggle ? "Tick the components to remove. " : ""}
          Maps show each component's weight at the electrodes; spectra its activity's power.
          {evidence.shown < evidence.n_components ? ` The first ${evidence.shown} of ${evidence.n_components} are shown.` : ""}
        </div>
        <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))" }}>
          {evidence.components.map((c) => {
            const values = layout ? layout.names.map((n) => c.weights[evidence.channels.indexOf(n)] ?? 0) : undefined;
            const on = chosen.has(String(c.index));
            return (
              <div key={c.index} className={`card compact${on ? " selected" : ""}`} style={{ gap: 6 }}>
                <label className="check">
                  <input type="checkbox" checked={on} disabled={!onToggle} onChange={() => onToggle?.(String(c.index))} />
                  <span>
                    <b>IC{c.index}</b> · {c.label}
                    {c.probability != null ? ` (${Math.round(c.probability * 100)}%)` : ""}
                    {c.explained_variance != null ? (
                      <span className="tiny muted"> · {(c.explained_variance * 100).toFixed(1)}% of variance</span>
                    ) : null}
                  </span>
                </label>
                <div className="row" style={{ gap: 8, flexWrap: "nowrap", alignItems: "center" }}>
                  {layout && values ? <HeadMap layout={layout} values={values} size={110} labelled={[]} /> : null}
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <LineChart
                      x={evidence.freqs}
                      series={[{ label: "power", values: c.psd_db, color: on ? TRACE : "var(--accent)" }]}
                      xLabel="Hz"
                      yLabel="dB"
                      height={150}
                      width={300}
                    />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  }
  return <pre className="log">{JSON.stringify(proposal.proposal, null, 2)}</pre>;
}

function medianOf(values: number[]): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}
