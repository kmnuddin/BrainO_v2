/** One subject: its recordings and its pipelines' steps. The trace browser will open here. */

import { Loading, Pill } from "../components/ui";
import { duration, plural } from "../lib/format";
import { useApp } from "../state/app";
import { SUBJECT_STATUS, useDataset } from "../state/dataset";

export function Subject({ subject }: { subject: string }) {
  const { go } = useApp();
  const info = useDataset();
  if (!info.summary) return <Loading />;
  const recordings = info.summary.recordings.filter((r) => r.subject === subject);
  const status = SUBJECT_STATUS[info.subjectStatus(subject)];
  const waiting = info.pending.filter((p) => p.arguments.subject === subject);

  return (
    <main className="workspace">
      <div className="row">
        <h1 className="mono">sub-{subject}</h1>
        <span className="row small" style={{ gap: 6 }}>
          <span className="dot" style={{ background: status.color }} />
          {status.label}
        </span>
        <span className="spacer" />
        {waiting.length ? (
          <button type="button" className="btn" onClick={() => go({ name: "decisions" })}>
            {plural(waiting.length, "decision")} waiting
          </button>
        ) : null}
      </div>

      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr>
              <th>Recording</th>
              <th>Length</th>
              <th>Channels</th>
              <th>Sampling</th>
              <th>Events</th>
              <th>Bad channels</th>
            </tr>
          </thead>
          <tbody>
            {recordings.map((r) => (
              <tr key={r.path}>
                <td className="mono">{r.path}</td>
                <td>{duration(r.duration_s)}</td>
                <td>
                  {Object.entries(r.channels)
                    .map(([type, n]) => `${n} ${type.toUpperCase()}`)
                    .join(", ")}
                </td>
                <td>{r.sampling_rate_hz ? `${r.sampling_rate_hz} Hz` : ""}</td>
                <td className="small">
                  {Object.entries(r.events)
                    .slice(0, 6)
                    .map(([type, n]) => `${type} (${n})`)
                    .join(", ")}
                  {Object.keys(r.events).length > 6 ? ", …" : ""}
                </td>
                <td>{r.bad_channels.length ? <Pill tone="warn">{r.bad_channels.join(", ")}</Pill> : <span className="muted">None</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {info.pipelines.map(({ item, state }) => {
        const steps = state?.subjects[subject] ?? {};
        return (
          <section key={item.name} className="card compact">
            <h2>{item.name}</h2>
            <div className="chips">
              {(state?.steps ?? item.steps?.map((s) => s.id) ?? []).map((step) => {
                const cell = steps[step];
                const tone = !cell ? "todo" : cell.status === "waiting" ? "warn" : cell.status === "rejected" ? "danger" : "ok";
                return (
                  <Pill key={step} tone={tone}>
                    {step}: {!cell ? "not run" : cell.status === "waiting" ? "decide" : cell.status}
                  </Pill>
                );
              })}
            </div>
          </section>
        );
      })}
      <p className="small muted">The trace browser for this subject's recordings will open here.</p>
    </main>
  );
}
