/** Pipeline specs in the app: building them step by step and reading the engine's checks. */

import type { CheckResult, PipelineSpec, PipelineStep, ToolInfo } from "../engine/types";

/** Arguments the runner sets (braino.pipelines.spec); never shown in a step's form. */
export const RUNNER_FIELDS = ["subject", "task", "session", "run"] as const;
export const GROUP_RUNNER_FIELDS = ["task", "session", "subjects", "run"] as const;

export const NAME_PATTERN = /^[a-z0-9][a-z0-9_-]*$/;
export const STEP_ID_PATTERN = /^[a-z][a-z0-9_]*$/;

export type StepGroup = "clean" | "measure" | "test";

/** Where a tool goes in the builder's "Add a step" menu; null for tools that are not steps. */
export function stepGroup(tool: ToolInfo): StepGroup | null {
  const n = tool.name;
  if (n.startsWith("stats.")) return "test";
  if (["eeg.preprocess", "eeg.detect_bad_channels", "eeg.ica_clean"].includes(n)) return "clean";
  if (n.startsWith("eeg.")) return "measure";
  return null;
}

export function emptySpec(task: string): PipelineSpec {
  return {
    name: "my-pipeline",
    description: "",
    task,
    session: null,
    run: null,
    subjects: null,
    exclude_subjects: [],
    steps: [],
    group: [],
  };
}

/** A step id for ``tool`` not used yet: "preprocess", "preprocess_2", ... */
export function newStepId(spec: PipelineSpec, tool: string): string {
  const base = (tool.split(".").pop() ?? "step").replace(/[^a-z0-9_]/g, "_").replace(/^[^a-z]+/, "") || "step";
  const used = new Set([...spec.steps, ...spec.group].map((s) => s.id));
  if (!used.has(base)) return base;
  for (let i = 2; ; i++) if (!used.has(`${base}_${i}`)) return `${base}_${i}`;
}

/** The earlier step whose output a step reads, as the engine's plan check sees it
 * (braino.pipelines.checks). Null when it reads the recording itself; "missing" text when the
 * step it needs is not there. */
export function inputOf(
  spec: PipelineSpec,
  step: PipelineStep,
  group: boolean,
): { step: PipelineStep | null; needs?: string } {
  if (group) {
    const data = String(step.args.data ?? "erp");
    const tool = { erp: "eeg.erp", psd: "eeg.spectral", tfr: "eeg.tfr" }[data];
    if (!tool) return { step: null };
    const source = [...spec.steps].reverse().find((s) => s.tool === tool);
    return source ? { step: source } : { step: null, needs: tool };
  }
  const index = spec.steps.indexOf(step);
  const earlier = spec.steps.slice(0, index).reverse();
  const tool = step.tool;
  if (tool === "eeg.preprocess" || tool === "eeg.qc_report") return { step: null };
  const readsEpochs = tool === "eeg.erp" || tool === "eeg.tfr" || (tool === "eeg.spectral" && step.args.source === "epochs");
  if (readsEpochs) {
    const epochs = earlier.find((s) => s.tool === "eeg.epoch");
    return epochs ? { step: epochs } : { step: null, needs: "eeg.epoch" };
  }
  const cleaned = earlier.find((s) =>
    ["eeg.preprocess", "eeg.detect_bad_channels", "eeg.ica_clean"].includes(s.tool),
  );
  return cleaned ? { step: cleaned } : { step: null, needs: "eeg.preprocess" };
}

export interface StepProblems {
  /** Problems of the whole step, in words. */
  messages: string[];
  /** Problems of one setting, by field name. */
  fields: Record<string, string>;
}

/** The engine's check, by step id ("" for the pipeline as a whole). Problems are the spec's
 * own errors ("step 'x' (tool): field: message; ..."), findings come from the plan check. */
export function problemsByStep(check: CheckResult | null): Map<string, StepProblems> {
  const out = new Map<string, StepProblems>();
  const get = (id: string) => {
    let entry = out.get(id);
    if (!entry) {
      entry = { messages: [], fields: {} };
      out.set(id, entry);
    }
    return entry;
  };
  if (!check) return out;
  for (const line of check.problems) {
    const match = line.match(/^(?:group )?step '([^']+)'(?: \(([^)]+)\))?: (.*)$/);
    if (!match) {
      if (line.trim()) get("").messages.push(line.trim());
      continue;
    }
    const [, id, tool, rest] = match;
    const entry = get(id);
    if (!tool) {
      entry.messages.push(rest);
      continue;
    }
    for (const part of rest.split("; ")) {
      const field = part.match(/^([a-z_0-9.]+): (.*)$/);
      if (field) entry.fields[field[1].split(".")[0]] = field[2];
      else entry.messages.push(part);
    }
  }
  for (const finding of check.findings) {
    if (finding.severity !== "error" && finding.severity !== "warning") continue;
    const text = finding.severity === "warning" ? `Warning: ${finding.message}` : finding.message;
    get(finding.step ?? "").messages.push(text);
  }
  return out;
}

/** Steps moved: ``index`` by ``by`` places (within the list). */
export function moved<T>(items: T[], index: number, by: number): T[] {
  const to = index + by;
  if (to < 0 || to >= items.length) return items;
  const next = [...items];
  const [item] = next.splice(index, 1);
  next.splice(to, 0, item);
  return next;
}

/** Whether two specs are the same (key order aside). */
export function sameSpec(a: PipelineSpec | null, b: PipelineSpec | null): boolean {
  return JSON.stringify(normalise(a)) === JSON.stringify(normalise(b));
}

function normalise(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(normalise);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([k, v]) => [k, normalise(v)]),
    );
  }
  return value;
}
