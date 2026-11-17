/** This session's jobs, kept up to date from the engine's events. */

import type { EngineMessage, Job } from "../engine/types";

export type JobEventListener = (jobId: string, event: Record<string, unknown>) => void;

export class JobStore {
  private jobs = new Map<string, Job>();
  private events = new Map<string, Record<string, unknown>[]>();
  private listeners = new Set<() => void>();
  private eventListeners = new Set<JobEventListener>();
  private snapshot: Job[] = [];

  /** Apply one message from the engine. Unknown messages are ignored. */
  apply(message: unknown): void {
    const m = message as EngineMessage;
    if (!m || typeof m !== "object") return;
    if (m.type === "jobs" && Array.isArray(m.jobs)) {
      this.jobs = new Map(m.jobs.map((j) => [j.id, j]));
      this.changed();
    } else if (m.type === "job" && m.job) {
      this.jobs.set(m.job.id, { ...this.jobs.get(m.job.id), ...m.job });
      this.changed();
    } else if (m.type === "job_event" && m.event) {
      const list = this.events.get(m.job_id) ?? [];
      list.push(m.event);
      this.events.set(m.job_id, list);
      for (const listener of this.eventListeners) listener(m.job_id, m.event);
      this.changed();
    }
  }

  /** Every job, newest first. The same array until something changes. */
  list = (): Job[] => this.snapshot;

  get(id: string): Job | undefined {
    return this.jobs.get(id);
  }

  /** What the job reported so far (pipeline steps, imported recordings). */
  eventsOf(id: string): Record<string, unknown>[] {
    return this.events.get(id) ?? [];
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  onJobEvent(listener: JobEventListener): () => void {
    this.eventListeners.add(listener);
    return () => this.eventListeners.delete(listener);
  }

  /** Resolves with the job once it has finished (done, failed or stopped). */
  finished(id: string): Promise<Job> {
    return new Promise((resolve) => {
      const check = () => {
        const job = this.jobs.get(id);
        if (job && !isActive(job)) {
          off();
          resolve(job);
        }
      };
      const off = this.subscribe(check);
      check();
    });
  }

  /** Forget everything (the engine restarted, so its jobs are gone). */
  reset(): void {
    this.jobs.clear();
    this.events.clear();
    this.changed();
  }

  private changed(): void {
    this.snapshot = [...this.jobs.values()].sort((a, b) =>
      b.created_at.localeCompare(a.created_at),
    );
    for (const listener of this.listeners) listener();
  }
}

export const isActive = (job: Job) => job.status === "queued" || job.status === "running";

/** Same folder, whatever the case and slashes (Windows paths). */
export function samePath(a: string | null | undefined, b: string | null | undefined): boolean {
  if (!a || !b) return false;
  const norm = (p: string) => p.replace(/[\\/]+$/, "").replace(/\\/g, "/").toLowerCase();
  return norm(a) === norm(b);
}
