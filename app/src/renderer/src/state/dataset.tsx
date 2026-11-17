/** What the screens of an open dataset share: its summary, its pipelines' state per subject,
 * and the decisions waiting. Reloaded when a job on the dataset reports or finishes. */

import { type ReactNode, createContext, useContext, useEffect, useMemo, useRef } from "react";
import { api } from "../engine/api";
import type {
  DatasetSummary,
  Job,
  PipelineListItem,
  PipelineState,
  Proposal,
} from "../engine/types";
import { useAsync } from "../lib/useAsync";
import { useApp, useDatasetJobs } from "./app";
import { isActive, samePath } from "./jobs";

export type SubjectStatus = "analysed" | "decision" | "running" | "imported";

export interface DatasetInfo {
  path: string;
  summary: DatasetSummary | undefined;
  summaryError: Error | undefined;
  /** Pipelines whose task the dataset has, with their state. */
  pipelines: { item: PipelineListItem; state: PipelineState | null }[];
  pending: Proposal[];
  jobs: Job[];
  /** The running pipeline job's name, if one runs. */
  runningPipeline: string | null;
  subjectStatus(subject: string): SubjectStatus;
  reload(): void;
}

const Context = createContext<DatasetInfo | null>(null);

export function useDataset(): DatasetInfo {
  const info = useContext(Context);
  if (!info) throw new Error("useDataset outside DatasetProvider");
  return info;
}

export function DatasetProvider({ path, children }: { path: string; children: ReactNode }) {
  const { datasetVersion, jobs: store } = useApp();
  const jobs = useDatasetJobs();

  const summary = useAsync(() => api.datasetSummary(path, 2000), [path, datasetVersion]);
  const pending = useAsync(() => api.proposals(path, "pending"), [path, datasetVersion]);
  const tasks = summary.data?.tasks.join("|") ?? "";
  const pipelines = useAsync(async () => {
    const [all, ran] = await Promise.all([api.pipelines(), api.datasetPipelines(path)]);
    const wanted = new Set(tasks.split("|"));
    const matching = all.filter((p) => !p.error && p.task && wanted.has(p.task));
    // Pipelines run from a template or the builder without being saved still have results.
    for (const item of ran) if (!matching.some((p) => p.name === item.name)) matching.push(item);
    return Promise.all(
      matching.map(async (item) => ({
        item,
        state: await api.pipelineState(item.name, path).catch(() => null),
      })),
    );
  }, [path, tasks, datasetVersion]);

  // A pipeline job's step outcomes change the state: reload, at most twice a second.
  const reloadState = pipelines.reload;
  const reloadPending = pending.reload;
  const timer = useRef<number | undefined>(undefined);
  useEffect(
    () =>
      store.onJobEvent((jobId) => {
        const job = store.get(jobId);
        if (!job || !samePath(job.dataset, path) || job.kind !== "pipeline") return;
        if (timer.current !== undefined) return;
        timer.current = window.setTimeout(() => {
          timer.current = undefined;
          reloadState();
          reloadPending();
        }, 500);
      }),
    [store, path, reloadState, reloadPending],
  );

  const runningPipeline = useMemo(() => {
    const job = jobs.find((j) => j.kind === "pipeline" && isActive(j));
    return job ? job.title.replace(/^Pipeline /, "") : null;
  }, [jobs]);

  const value = useMemo<DatasetInfo>(() => {
    const states = (pipelines.data ?? []).map((p) => p.state).filter((s): s is PipelineState => !!s);
    const waiting = new Set<string>();
    for (const p of pending.data ?? []) {
      const subject = p.arguments.subject;
      if (typeof subject === "string") waiting.add(subject);
    }
    return {
      path,
      summary: summary.data,
      summaryError: summary.error,
      pipelines: pipelines.data ?? [],
      pending: pending.data ?? [],
      jobs,
      runningPipeline,
      subjectStatus(subject) {
        const records = states.map((s) => s.subjects[subject] ?? {});
        if (waiting.has(subject) || records.some((r) => Object.values(r).some((c) => c.status === "waiting"))) {
          return "decision";
        }
        if (runningPipeline) {
          const state = states.find((s) => s.name === runningPipeline);
          const done = state?.subjects[subject] ?? {};
          if (state && state.steps.some((step) => !done[step])) return "running";
        }
        const complete = states.some(
          (s) => s.steps.length > 0 && s.steps.every((step) => s.subjects[subject]?.[step]?.status === "done"),
        );
        return complete ? "analysed" : "imported";
      },
      reload() {
        summary.reload();
        pending.reload();
        pipelines.reload();
      },
    };
  }, [path, summary, pending, pipelines, jobs, runningPipeline]);

  return <Context.Provider value={value}>{children}</Context.Provider>;
}

export const SUBJECT_STATUS: Record<SubjectStatus, { color: string; label: string }> = {
  analysed: { color: "var(--ok)", label: "Analysed" },
  decision: { color: "var(--warn)", label: "Decision" },
  running: { color: "var(--accent)", label: "Running" },
  imported: { color: "var(--muted-2)", label: "Imported" },
};
