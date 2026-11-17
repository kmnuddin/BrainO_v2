/** What the engine service returns (see docs/reference/service.md). */

export type JobKind = "tool" | "pipeline" | "import" | "decision" | "model_setup" | "example";
export type JobStatus = "queued" | "running" | "done" | "failed" | "cancelled";

export interface Job {
  id: string;
  kind: JobKind;
  title: string;
  dataset: string | null;
  status: JobStatus;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  progress: number | null;
  message: string | null;
  run_id: string | null;
  error: string | null;
  /** Only in GET /api/jobs/{id}. */
  events?: Record<string, unknown>[];
  result?: unknown;
}

export type EngineMessage =
  | { type: "jobs"; jobs: Job[] }
  | { type: "job"; job: Job }
  | { type: "job_event"; job_id: string; event: Record<string, unknown> };

export interface Gpu {
  name: string;
  vendor: string;
  memory_gb: number;
  unified: boolean;
}

export interface Hardware {
  platform: string;
  os: string;
  arch: string;
  cpu: string | null;
  cpu_count: number | null;
  memory_gb: number | null;
  gpus: Gpu[];
}

export interface Recommendation {
  tier: string;
  model: string;
  placement: "cpu" | "gpu" | "partial";
  backend: string;
  gpu: string | null;
  reasons: string[];
  warnings: string[];
}

export interface ModelSettings {
  model: string | null;
  backend: string | null;
  placement: "cpu" | "gpu" | "partial" | null;
  tier: string | null;
}

export interface SystemInfo {
  version: string;
  hardware: Hardware;
  jobs: { max_parallel: number; threads_per_job: number };
  model: ModelSettings;
  installed_models: string[];
  recommendation: Recommendation;
  models: { id: string; name: string; size_gb: number }[];
  user_folder: string;
}

export interface RecordingSummary {
  subject: string;
  session: string | null;
  task: string | null;
  run: string | null;
  datatype: string;
  path: string;
  sampling_rate_hz: number | null;
  duration_s: number | null;
  line_freq_hz: number | null;
  channels: Record<string, number>;
  bad_channels: string[];
  events: Record<string, number>;
}

export interface DatasetSummary {
  name: string | null;
  bids_version: string | null;
  n_subjects: number;
  subjects: string[];
  sessions: string[];
  tasks: string[];
  datatypes: string[];
  n_recordings: number;
  total_duration_s: number;
  recordings: RecordingSummary[];
  truncated: boolean;
}

export interface DetectedRecording {
  path: string;
  format: string;
  format_name: string;
  files: string[];
}

export interface DetectResult {
  recordings: DetectedRecording[];
  skipped: { path: string; reason: string }[];
}

export interface ImportItem {
  source: string;
  subject_id: string;
  task: string;
  session?: string | null;
  run?: number | null;
  line_freq?: number | null;
}

export interface ImportResultItem {
  index: number;
  ok: boolean;
  subject?: string;
  format?: string;
  check_by_hand?: string[];
  error?: string;
}

export interface PipelineStepRef {
  id: string;
  tool: string;
}

export interface PipelineListItem {
  name: string;
  /** "ran": only run on a dataset (from a template or the builder), not saved. */
  source: "builtin" | "user" | "ran";
  description?: string;
  task?: string;
  steps?: PipelineStepRef[];
  group?: PipelineStepRef[];
  error?: string;
}

export interface PipelineStep {
  id: string;
  tool: string;
  args: Record<string, unknown>;
}

export interface PipelineSpec {
  name: string;
  description: string;
  task: string;
  session: string | null;
  run: string | null;
  subjects: string[] | null;
  exclude_subjects: string[];
  steps: PipelineStep[];
  group: PipelineStep[];
}

export type StepState = "done" | "waiting" | "rejected" | "approved";

export interface PipelineState {
  name: string;
  steps: string[];
  group_steps: string[];
  subjects: Record<
    string,
    Record<string, { tool: string; status: StepState; proposal_id: string | null; finished_at: string }>
  >;
}

export type OutcomeStatus = "ran" | "cached" | "pending" | "rejected" | "failed" | "blocked";

export interface StepOutcome {
  subject: string;
  step: string;
  tool: string;
  status: OutcomeStatus;
  proposal_id: string | null;
  message: string | null;
}

export interface Decision {
  approved: boolean;
  decided_by: string;
  decided_at: string;
  note: string | null;
  final: Record<string, unknown> | null;
  edited: boolean;
  applied_in: string | null;
  apply_result: Record<string, unknown> | null;
}

export interface Proposal {
  proposal_id: string;
  tool: string;
  run_id: string;
  created_at: string;
  arguments: Record<string, unknown>;
  proposal: Record<string, unknown>;
  status: "pending" | "approved" | "rejected";
  decision: Decision | null;
}

export interface ElectrodeLayout {
  names: string[];
  x: number[];
  y: number[];
}

export type Evidence =
  | { kind: "none" }
  | {
      kind: "bad_channels";
      layout: ElectrodeLayout | null;
      amplitude_uv: Record<string, number>;
      proposed: string[];
      snippet: { start_s: number; sfreq: number };
      channels: {
        name: string;
        reasons: string[];
        signal: number[];
        neighbour: string | null;
        neighbour_signal: number[];
      }[];
    }
  | {
      kind: "ica";
      layout: ElectrodeLayout | null;
      channels: string[];
      freqs: number[];
      components: {
        index: number;
        label: string;
        probability: number | null;
        explained_variance: number | null;
        proposed: boolean;
        weights: number[];
        psd_db: number[];
      }[];
      shown: number;
      n_components: number;
    };

export interface CheckFinding {
  severity: "error" | "warning" | "info";
  message: string;
  step?: string | null;
  [key: string]: unknown;
}

export interface CheckResult {
  valid: boolean;
  problems: string[];
  findings: CheckFinding[];
}

export type JsonSchema = {
  type?: string | string[];
  title?: string;
  description?: string;
  default?: unknown;
  enum?: unknown[];
  const?: unknown;
  items?: JsonSchema;
  properties?: Record<string, JsonSchema>;
  required?: string[];
  anyOf?: JsonSchema[];
  oneOf?: JsonSchema[];
  allOf?: JsonSchema[];
  $ref?: string;
  $defs?: Record<string, JsonSchema>;
  additionalProperties?: boolean | JsonSchema;
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
  minItems?: number;
  maxItems?: number;
  prefixItems?: JsonSchema[];
  examples?: unknown[];
};

export interface ResultStep {
  id: string;
  tool: string;
  kind: "erp" | "psd" | "tfr";
  subjects: string[];
  names: string[];
  windows: { name: string | null; tmin: number | null; tmax: number | null; channels: string[] | null }[];
}

export interface Cluster {
  p: number;
  mass: number;
  sign: "positive" | "negative";
  channels: string[];
  ranges: Record<string, [number, number]>;
  n_points: number;
}

export interface GroupResult {
  id: string;
  tool: string;
  args: Record<string, unknown>;
  /** The stats tool's output: {result: {...}, table, notes}. */
  result: { result?: Record<string, unknown> & { clusters?: Cluster[] }; notes?: string[]; table?: string } | null;
  finished_at: string | null;
}

export interface ResultsIndex {
  name: string;
  steps: ResultStep[];
  measures: {
    subject: string;
    step: string;
    measure: string;
    waveform: string;
    kind: string;
    channels: string[];
    value_uv: number;
    latency_ms: number | null;
    n_trials: number | null;
  }[];
  group: GroupResult[];
}

export type ResultData =
  | {
      kind: "erp" | "psd";
      subjects: string[];
      channels: string[];
      channel: string;
      x: number[];
      x_unit: string;
      unit: string;
      series: Record<string, number[]>;
      layout?: ElectrodeLayout | null;
      maps?: Record<string, number[]>;
      window?: [number, number] | null;
    }
  | {
      kind: "tfr";
      subjects: string[];
      channels: string[];
      channel: string;
      names: string[];
      name: string;
      times: number[];
      freqs: number[];
      unit: string;
      power: number[][];
    };

export interface TemplateInfo {
  name: string;
  description: string;
  settings_schema: JsonSchema;
}

export interface ToolInfo {
  name: string;
  description: string;
  risk: "read" | "compute" | "decision";
  needs_dataset: boolean;
  per_recording: boolean;
  input_schema: JsonSchema;
  output_schema: JsonSchema;
}
