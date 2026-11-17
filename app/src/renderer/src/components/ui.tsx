/** Small pieces shared by the screens. */

import type { ReactNode } from "react";
import { api } from "../engine/api";
import type { Job } from "../engine/types";
import { percent } from "../lib/format";

export function ProgressBar({
  value,
  thin,
  label,
}: {
  value: number | null | undefined;
  thin?: boolean;
  label?: string;
}) {
  const known = value != null;
  return (
    <div
      className={`progress${thin ? " thin" : ""}${known ? "" : " indeterminate"}`}
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={known ? Math.round(value * 100) : undefined}
    >
      <div style={{ width: `${Math.round((value ?? 0) * 100)}%` }} />
    </div>
  );
}

export type Tone = "ok" | "warn" | "info" | "todo" | "danger";

export function Pill({ tone, children }: { tone: Tone; children: ReactNode }) {
  return <span className={`pill ${tone}`}>{children}</span>;
}

export function ErrorNotice({ error, title }: { error: unknown; title?: string }) {
  if (!error) return null;
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div className="notice error" role="alert">
      {title ? <strong>{title}: </strong> : null}
      {message}
    </div>
  );
}

export function Loading({ what = "Loading" }: { what?: string }) {
  return <div className="empty">{what}…</div>;
}

/** A job's state in one line, with its progress and a Cancel button while it runs. */
export function JobLine({ job, cancellable = true }: { job: Job; cancellable?: boolean }) {
  const running = job.status === "running" || job.status === "queued";
  return (
    <div className="stack" style={{ gap: 6 }}>
      <div className="row" style={{ gap: 8 }}>
        <span style={{ fontWeight: 600 }}>{job.title}</span>
        <JobStatusPill job={job} />
        <span className="spacer" />
        {running && job.progress != null ? <span className="small muted">{percent(job.progress)}</span> : null}
        {running && cancellable ? (
          <button type="button" className="btn small" onClick={() => void api.cancelJob(job.id)}>
            {job.status === "queued" ? "Remove from queue" : "Stop"}
          </button>
        ) : null}
      </div>
      {running ? <ProgressBar value={job.status === "queued" ? 0 : job.progress} thin label={job.title} /> : null}
      {job.message && running ? <span className="small muted">{job.message}</span> : null}
      {job.error ? <ErrorNotice error={job.error} /> : null}
    </div>
  );
}

export function JobStatusPill({ job }: { job: Job }) {
  switch (job.status) {
    case "queued":
      return <Pill tone="todo">Waiting for a free slot</Pill>;
    case "running":
      return <Pill tone="info">{job.message === "cancelling" ? "Stopping" : "Running"}</Pill>;
    case "done":
      return <Pill tone="ok">Done</Pill>;
    case "failed":
      return <Pill tone="danger">Failed</Pill>;
    case "cancelled":
      return <Pill tone="todo">Stopped</Pill>;
  }
}
