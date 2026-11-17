/** Subject, session, task and run read from recording file names, and mapping CSVs. */

export interface Guess {
  subject: string;
  session: string;
  task: string;
  run: string;
}

const STEM_EXTENSIONS = /\.(edf|bdf|vhdr|vmrk|eeg|fif|fif\.gz|set|fdt|cnt|mff|gdf|cdt|dap|dat|rs3|lay|21e|eeg|xdf)$/i;

export function stem(path: string): string {
  const name = path.replace(/[\\/]+$/, "").split(/[\\/]/).pop() ?? path;
  return name.replace(STEM_EXTENSIONS, "");
}

// Longer prefixes first: with "ses" before "sess", "sess1" would be session "s1".
const SUBJECT = [
  /(?:^|[_\-\s.])(?:subject|subj|sub|participant|part|pp|p|s)[-_\s]?([a-z]*\d+[a-z0-9]*)(?=$|[_\-\s.])/i,
];
const SESSION = /(?:^|[_\-\s.])(?:session|sess|ses|visit|v)[-_\s]?([a-z0-9]+)(?=$|[_\-\s.])/i;
const TASK = /(?:^|[_\-\s.])task[-_\s]?([a-z0-9]+)(?=$|[_\-\s.])/i;
const RUN = /(?:^|[_\-\s.])(?:run|r)[-_\s]?(\d+)(?=$|[_\-\s.])/i;

/** What a file name tells: ``s41_sess1.bdf`` is subject 41, session 1. Empty where it says
 * nothing. A name that is only a number (``017.edf``) is taken as the subject. */
export function guessFromName(path: string): Guess {
  const name = stem(path);
  const guess: Guess = { subject: "", session: "", task: "", run: "" };
  for (const pattern of SUBJECT) {
    const match = name.match(pattern);
    if (match) {
      guess.subject = match[1];
      break;
    }
  }
  if (!guess.subject && /^\d+$/.test(name)) guess.subject = name;
  guess.session = name.match(SESSION)?.[1] ?? "";
  guess.task = name.match(TASK)?.[1] ?? "";
  guess.run = name.match(RUN)?.[1] ?? "";
  return guess;
}

/** BIDS labels allow letters and digits only. */
export function cleanLabel(value: string): string {
  return value.replace(/[^A-Za-z0-9]/g, "");
}

export interface MappingRow {
  source: string;
  subject_id: string;
  task: string;
  session: string;
  run: string;
  line_freq: string;
}

/** Parse a mapping CSV (the columns of ``braino import --mapping``). Quotes are honoured. */
export function parseMappingCsv(text: string): MappingRow[] {
  const rows = parseCsv(text.replace(/^﻿/, ""));
  if (rows.length === 0) return [];
  const header = rows[0].map((h) => h.trim().toLowerCase());
  const missing = ["source", "subject_id", "task"].filter((c) => !header.includes(c));
  if (missing.length) throw new Error(`the mapping lacks the columns: ${missing.join(", ")}`);
  const column = (row: string[], name: string) => {
    const index = header.indexOf(name);
    return index >= 0 ? (row[index] ?? "").trim() : "";
  };
  return rows
    .slice(1)
    .filter((row) => row.some((cell) => cell.trim()))
    .map((row) => ({
      source: column(row, "source"),
      subject_id: column(row, "subject_id"),
      task: column(row, "task"),
      session: column(row, "session"),
      run: column(row, "run"),
      line_freq: column(row, "line_freq"),
    }));
}

function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') {
        cell += '"';
        i++;
      } else if (c === '"') {
        quoted = false;
      } else {
        cell += c;
      }
    } else if (c === '"') {
      quoted = true;
    } else if (c === ",") {
      row.push(cell);
      cell = "";
    } else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += c;
    }
  }
  if (cell || row.length) {
    row.push(cell);
    rows.push(row);
  }
  return rows;
}

/** The mapping row for a recording: the same path, or else the same file name. */
export function findMapping(rows: MappingRow[], path: string): MappingRow | undefined {
  const norm = (p: string) => p.replace(/\\/g, "/").toLowerCase();
  const exact = rows.find((r) => norm(r.source) === norm(path));
  if (exact) return exact;
  const name = norm(path).split("/").pop();
  return rows.find((r) => norm(r.source).split("/").pop() === name);
}
