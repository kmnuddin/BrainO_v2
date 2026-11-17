/** Numbers, sizes, times and paths in words. */

export function gigabytes(bytes: number | null | undefined, digits = 1): string {
  if (bytes == null) return "unknown";
  return `${(bytes / 2 ** 30).toFixed(digits)} GB`;
}

export function duration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "";
  if (seconds < 60) return `${Math.round(seconds)} s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min`;
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return rest ? `${hours} h ${rest} min` : `${hours} h`;
}

export function plural(n: number, one: string, many = `${one}s`): string {
  return `${n} ${n === 1 ? one : many}`;
}

/** "Today", "Yesterday", "3 days ago", "Last week", or the date. */
export function whenOpened(iso: string, now = new Date()): string {
  const then = new Date(iso);
  if (Number.isNaN(then.getTime())) return "";
  const day = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const days = Math.round((day(now) - day(then)) / 86_400_000);
  if (days <= 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 7) return `${days} days ago`;
  if (days < 14) return "Last week";
  return then.toLocaleDateString();
}

export function basename(path: string): string {
  const parts = path.replace(/[\\/]+$/, "").split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

export function dirname(path: string): string {
  const trimmed = path.replace(/[\\/]+$/, "");
  const index = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  return index > 0 ? trimmed.slice(0, index) : trimmed;
}

/** ``folder`` joined with ``name``, with the folder's own separator. */
export function joinPath(folder: string, ...names: string[]): string {
  const sep = folder.includes("\\") ? "\\" : "/";
  return [folder.replace(/[\\/]+$/, ""), ...names].join(sep);
}

/** A dataset's name: its own, unless that is missing or MNE-BIDS's placeholder, else its
 * folder's. */
export function datasetName(name: string | null | undefined, path: string): string {
  return name && name !== "[Unspecified]" ? name : basename(path);
}

/** "BrainO Qwen3.5 4B, fine-tuned for BrainO's tools (Q4_K_M)" → "BrainO Qwen3.5 4B". */
export function shortModelName(name: string): string {
  return name.split(",")[0].replace(/\s*\([^)]*\)\s*$/, "").trim();
}

export function percent(fraction: number | null | undefined): string {
  return fraction == null ? "" : `${Math.round(fraction * 100)}%`;
}
