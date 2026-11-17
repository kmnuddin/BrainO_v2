/** Calls to the engine service, through the main process. */

import type { HttpMethod } from "../../../shared/ipc";
import type {
  CheckResult,
  DatasetSummary,
  DetectResult,
  Evidence,
  ImportItem,
  Job,
  PipelineListItem,
  PipelineSpec,
  PipelineState,
  Proposal,
  ResultData,
  ResultsIndex,
  SystemInfo,
  TemplateInfo,
  ToolInfo,
} from "./types";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: unknown,
  ) {
    super(describeDetail(status, detail));
  }
}

/** An engine error in words: its message, or each validation error on a line. */
export function describeDetail(status: number, detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((e) => {
        const item = e as { loc?: unknown[]; msg?: string };
        const where = (item.loc ?? []).filter((p) => p !== "body").join(".");
        return where ? `${where}: ${item.msg}` : String(item.msg ?? e);
      })
      .join("\n");
  }
  if (detail && typeof detail === "object" && "message" in detail) {
    return String((detail as { message: unknown }).message);
  }
  return `the engine answered ${status}`;
}

async function call<T>(method: HttpMethod, path: string, body?: unknown): Promise<T> {
  const response = await window.braino.request<T | { detail: unknown }>(method, path, body);
  if (!response.ok) {
    const detail =
      response.data && typeof response.data === "object" && "detail" in response.data
        ? (response.data as { detail: unknown }).detail
        : response.data;
    throw new ApiError(response.status, detail);
  }
  return response.data as T;
}

const q = (params: Record<string, string | number | undefined | null>) => {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null) search.set(key, String(value));
  }
  return search.toString();
};

export const api = {
  system: () => call<SystemInfo>("GET", "/api/system"),
  tools: () => call<ToolInfo[]>("GET", "/api/tools"),
  datasetSummary: (path: string, maxRecordings?: number) =>
    call<DatasetSummary>(
      "GET",
      `/api/datasets/summary?${q({ path, max_recordings: maxRecordings })}`,
    ),
  detect: (paths: string[]) => call<DetectResult>("POST", "/api/import/detect", { paths }),
  importRecordings: (dataset: string, key: string, jobs: ImportItem[]) =>
    call<Job>("POST", "/api/jobs/import", { dataset, key, jobs }),
  example: (task: string, subjects: number) =>
    call<Job>("POST", "/api/jobs/example", { task, subjects }),
  modelSetup: (model: string | null, backend: string | null) =>
    call<Job>("POST", "/api/jobs/model-setup", { model, backend, benchmark: true }),
  runTool: (tool: string, args: Record<string, unknown>, dataset: string | null) =>
    call<Job>("POST", "/api/jobs/tool", { tool, arguments: args, dataset }),
  proposals: (dataset: string, status?: Proposal["status"]) =>
    call<Proposal[]>("GET", `/api/proposals?${q({ dataset, status })}`),
  proposal: (dataset: string, id: string) =>
    call<Proposal>("GET", `/api/proposals/${encodeURIComponent(id)}?${q({ dataset })}`),
  evidence: (dataset: string, id: string) =>
    call<Evidence>("GET", `/api/proposals/${encodeURIComponent(id)}/evidence?${q({ dataset })}`),
  decide: (
    dataset: string,
    id: string,
    approve: boolean,
    edits?: Record<string, unknown> | null,
    note?: string | null,
  ) =>
    call<Job>("POST", `/api/proposals/${encodeURIComponent(id)}/decide`, {
      dataset,
      approve,
      edits: edits ?? null,
      note: note ?? null,
    }),
  pipelines: () => call<PipelineListItem[]>("GET", "/api/pipelines"),
  pipeline: (name: string) =>
    call<{ source: "builtin" | "user"; spec: PipelineSpec }>(
      "GET",
      `/api/pipelines/${encodeURIComponent(name)}`,
    ),
  savePipeline: (spec: PipelineSpec) =>
    call<{ source: "user"; spec: PipelineSpec }>(
      "PUT",
      `/api/pipelines/${encodeURIComponent(spec.name)}`,
      { spec },
    ),
  deletePipeline: (name: string) =>
    call<{ deleted: string }>("DELETE", `/api/pipelines/${encodeURIComponent(name)}`),
  checkPipeline: (spec: PipelineSpec, dataset?: string | null, subjects?: string[] | null) =>
    call<CheckResult>("POST", "/api/pipelines/check", {
      spec,
      dataset: dataset ?? null,
      subjects: subjects ?? null,
    }),
  templates: () => call<TemplateInfo[]>("GET", "/api/templates"),
  buildTemplate: (name: string, settings: Record<string, unknown>, dataset: string) =>
    call<PipelineSpec>("POST", `/api/templates/${encodeURIComponent(name)}/build`, {
      settings,
      dataset,
    }),
  datasetPipelines: (dataset: string) =>
    call<PipelineListItem[]>("GET", `/api/datasets/pipelines?${q({ dataset })}`),
  results: (name: string, dataset: string) =>
    call<ResultsIndex>("GET", `/api/pipelines/${encodeURIComponent(name)}/results?${q({ dataset })}`),
  resultData: (
    name: string,
    step: string,
    params: {
      dataset: string;
      subject?: string | null;
      channel?: string | null;
      tmin?: number | null;
      tmax?: number | null;
      condition?: string | null;
    },
  ) =>
    call<ResultData>(
      "GET",
      `/api/pipelines/${encodeURIComponent(name)}/results/${encodeURIComponent(step)}?${q(params)}`,
    ),
  pipelineState: (name: string, dataset: string) =>
    call<PipelineState>(
      "GET",
      `/api/pipelines/${encodeURIComponent(name)}/state?${q({ dataset })}`,
    ),
  runPipeline: (body: {
    dataset: string;
    name?: string;
    spec?: PipelineSpec;
    subjects?: string[] | null;
    from_step?: string | null;
    auto_approve?: boolean;
  }) => call<Job>("POST", "/api/jobs/pipeline", body),
  job: (id: string) => call<Job>("GET", `/api/jobs/${encodeURIComponent(id)}`),
  cancelJob: (id: string) => call<Job>("POST", `/api/jobs/${encodeURIComponent(id)}/cancel`),
  runs: (dataset: string) => call<string[]>("GET", `/api/runs?${q({ dataset })}`),
  run: (dataset: string, id: string) =>
    call<Record<string, unknown>>("GET", `/api/runs/${encodeURIComponent(id)}?${q({ dataset })}`),
};
