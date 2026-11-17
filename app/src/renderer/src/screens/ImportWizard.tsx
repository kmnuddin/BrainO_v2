/** Import wizard: choose files → format → subjects and sessions → personal details → import.
 *
 * The person's own ID typed here goes only into the private subject key; the dataset gets a
 * neutral label (001, 002, ...). Source paths never reach the dataset or the job's results. */

import { type DragEvent, useMemo, useState } from "react";
import { ErrorNotice, JobLine, Pill } from "../components/ui";
import { api } from "../engine/api";
import type { DetectedRecording, ImportItem, ImportResultItem, Job } from "../engine/types";
import {
  type MappingRow,
  cleanLabel,
  findMapping,
  guessFromName,
  parseMappingCsv,
  stem,
} from "../lib/filenames";
import { basename, joinPath, plural } from "../lib/format";
import { useApp, useJobs } from "../state/app";
import { isActive } from "../state/jobs";

const STEPS = ["Choose files", "Format", "Subjects and sessions", "Personal details", "Import"];

interface Row {
  recording: DetectedRecording;
  subject: string;
  session: string;
  task: string;
  run: string;
  /** Filled in from the file name or a mapping, or left for the user. */
  filled: boolean;
}

export function ImportWizard() {
  const { dataset, settings, system, openDataset, jobs: store } = useApp();
  const jobs = useJobs();
  const [step, setStep] = useState(0);
  const [rows, setRows] = useState<Row[]>([]);
  const [skipped, setSkipped] = useState<{ path: string; reason: string }[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [lineFreq, setLineFreq] = useState<"" | "50" | "60">("");
  const [name, setName] = useState("");
  const [keyPath, setKeyPath] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [results, setResults] = useState<ImportResultItem[] | null>(null);

  const target = dataset ?? (name && settings ? joinPath(settings.datasetsFolder, name) : "");
  const defaultKey =
    system && target ? joinPath(system.user_folder, "keys", `${basename(target)}.subject-key.json`) : "";
  const key = keyPath || defaultKey;
  const job: Job | undefined = jobs.find((j) => j.id === jobId);
  const importing = !!job && isActive(job);

  // --- step 1: files -----------------------------------------------------------------------

  const add = async (paths: string[]) => {
    if (paths.length === 0) return;
    setBusy(true);
    setError(null);
    try {
      const found = await api.detect(paths);
      setRows((current) => {
        const known = new Set(current.map((r) => r.recording.path.toLowerCase()));
        const fresh = found.recordings
          .filter((r) => !known.has(r.path.toLowerCase()))
          .map((recording) => {
            const guess = guessFromName(recording.path);
            return {
              recording,
              subject: guess.subject,
              session: cleanLabel(guess.session),
              task: cleanLabel(guess.task),
              run: guess.run,
              filled: !!guess.subject,
            };
          });
        return [...current, ...fresh];
      });
      setSkipped((current) => [...current, ...found.skipped]);
      if (found.recordings.length > 0 && step === 0) setStep(1);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  const chooseFiles = async (kind: "files" | "folders") =>
    add(await window.braino.choose({ kind, title: kind === "files" ? "Choose recordings" : "Choose folders" }));

  // --- step 3: mapping and validation ------------------------------------------------------

  const loadMapping = async () => {
    const [file] = await window.braino.choose({
      kind: "file",
      title: "Load a mapping CSV",
      filters: [{ name: "CSV", extensions: ["csv"] }],
    });
    if (!file) return;
    setError(null);
    try {
      const mapping: MappingRow[] = parseMappingCsv(await window.braino.readTextFile(file));
      let matched = 0;
      setRows((current) =>
        current.map((row) => {
          const found = findMapping(mapping, row.recording.path);
          if (!found) return row;
          matched++;
          return {
            ...row,
            subject: found.subject_id,
            task: cleanLabel(found.task) || row.task,
            session: cleanLabel(found.session),
            run: found.run,
            filled: true,
          };
        }),
      );
      if (matched === 0) setError("No row of the mapping names one of these recordings.");
    } catch (e) {
      setError(e);
    }
  };

  const problems = useMemo(() => rowProblems(rows), [rows]);
  const filled = rows.filter((r) => r.filled).length;
  const formats = useMemo(() => {
    const counts = new Map<string, number>();
    for (const r of rows) counts.set(r.recording.format_name, (counts.get(r.recording.format_name) ?? 0) + 1);
    return [...counts.entries()];
  }, [rows]);

  const update = (index: number, changes: Partial<Row>) =>
    setRows((current) => current.map((r, i) => (i === index ? { ...r, ...changes } : r)));
  const setTaskForAll = (task: string) => setRows((current) => current.map((r) => ({ ...r, task })));

  // --- step 5: import ----------------------------------------------------------------------

  const start = async () => {
    setError(null);
    setResults(null);
    const items: ImportItem[] = rows.map((r) => ({
      source: r.recording.path,
      subject_id: r.subject.trim(),
      task: r.task,
      session: r.session || null,
      run: r.run ? Number(r.run) : null,
      line_freq: lineFreq ? Number(lineFreq) : null,
    }));
    try {
      const queued = await api.importRecordings(target, key, items);
      setJobId(queued.id);
      setStep(4);
      const finished = await store.finished(queued.id);
      if (finished.status === "done") {
        const full = await api.job(queued.id);
        setResults((full.result as { results: ImportResultItem[] }).results);
      }
    } catch (e) {
      setError(e);
    }
  };

  const canNext = [
    rows.length > 0,
    rows.length > 0,
    rows.length > 0 && problems.size === 0,
    !!target && !!key && (dataset !== null || /^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(name)),
    false,
  ][step];

  return (
    <main className="workspace">
      <div className="row" style={{ alignItems: "baseline" }}>
        <h1>Import recordings</h1>
        <span className="muted">{dataset ? `into ${basename(dataset)}` : "into a new dataset"}</span>
      </div>

      <ol className="steps" aria-label="Steps">
        {STEPS.map((label, i) => (
          <li key={label} className={i < step ? "done" : i === step ? "current" : ""} aria-current={i === step ? "step" : undefined}>
            {i + 1} {label}
            {i < step ? " ✓" : ""}
          </li>
        ))}
      </ol>

      {step === 0 ? <ChooseStep busy={busy} onAdd={add} onChoose={chooseFiles} /> : null}

      {step === 1 ? (
        <section className="stack">
          <div className="card compact">
            <div className="tiny muted">Found</div>
            <div style={{ fontWeight: 600 }}>{plural(rows.length, "recording")}</div>
            <div className="small muted">{formats.map(([f, n]) => `${f} (${n})`).join(" · ")}</div>
          </div>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Recording</th>
                  <th>Format</th>
                  <th>Files</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={r.recording.path}>
                    <td className="mono break">{r.recording.path}</td>
                    <td>{r.recording.format_name}</td>
                    <td className="muted">{r.recording.files.length}</td>
                    <td>
                      <button type="button" className="btn small link" onClick={() => setRows(rows.filter((_, j) => j !== i))}>
                        Leave out
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {skipped.length > 0 ? (
            <details className="card compact">
              <summary style={{ cursor: "pointer" }}>{plural(skipped.length, "other file")} left out</summary>
              <div className="table-wrap" style={{ marginTop: 10 }}>
                <table className="table">
                  <tbody>
                    {skipped.map((s) => (
                      <tr key={s.path}>
                        <td className="mono break">{s.path}</td>
                        <td className="small muted">{s.reason}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          ) : null}
          <div className="row">
            <button type="button" className="btn" onClick={() => void chooseFiles("files")}>
              Add more files…
            </button>
          </div>
        </section>
      ) : null}

      {step === 2 ? (
        <section className="stack">
          <div className="row" style={{ alignItems: "stretch" }}>
            <div className="card compact" style={{ flex: "1 1 260px" }}>
              <div className="tiny muted">Filled in from the file names</div>
              <div style={{ fontWeight: 600 }}>
                {filled} of {rows.length} rows
              </div>
              {rows.length - filled > 0 ? (
                <div className="small" style={{ color: "var(--warn-text)" }}>
                  {plural(rows.length - filled, "row needs", "rows need")} you: the names don't follow a pattern
                </div>
              ) : null}
            </div>
            <div className="stack" style={{ flex: "1 1 220px", justifyContent: "center", gap: 8 }}>
              <button type="button" className="btn" onClick={() => void loadMapping()}>
                Load a mapping CSV…
              </button>
              <label className="field">
                Task for every row
                <input
                  className="text mono"
                  placeholder="e.g. oddball"
                  onChange={(e) => setTaskForAll(cleanLabel(e.target.value))}
                />
              </label>
            </div>
            <div className="stack" style={{ flex: "1 1 180px", justifyContent: "center" }}>
              <label className="field">
                Mains frequency
                <select className="text" value={lineFreq} onChange={(e) => setLineFreq(e.target.value as "" | "50" | "60")}>
                  <option value="">From the recordings</option>
                  <option value="50">50 Hz (Europe, Asia, Africa)</option>
                  <option value="60">60 Hz (Americas)</option>
                </select>
              </label>
            </div>
          </div>
          <div className="table-wrap">
            <table className="table" style={{ minWidth: 760 }}>
              <thead>
                <tr>
                  <th>File</th>
                  <th title="Kept only in the private subject key">Person's ID</th>
                  <th>Session</th>
                  <th>Task</th>
                  <th>Run</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => {
                  const issue = problems.get(i);
                  return (
                    <tr key={r.recording.path} className={r.subject ? "" : "attention"}>
                      <td className="mono">{stem(r.recording.path)}</td>
                      <td>
                        <input
                          aria-label={`Person's ID for ${stem(r.recording.path)}`}
                          className={`text cell${r.subject ? "" : " attention"}`}
                          style={{ width: 110 }}
                          value={r.subject}
                          placeholder="needed"
                          onChange={(e) => update(i, { subject: e.target.value, filled: true })}
                        />
                      </td>
                      <td>
                        <input
                          aria-label={`Session for ${stem(r.recording.path)}`}
                          className="text cell"
                          style={{ width: 70 }}
                          value={r.session}
                          onChange={(e) => update(i, { session: cleanLabel(e.target.value) })}
                        />
                      </td>
                      <td>
                        <input
                          aria-label={`Task for ${stem(r.recording.path)}`}
                          className={`text cell${r.task ? "" : " attention"}`}
                          style={{ width: 110 }}
                          value={r.task}
                          placeholder="needed"
                          onChange={(e) => update(i, { task: cleanLabel(e.target.value) })}
                        />
                      </td>
                      <td>
                        <input
                          aria-label={`Run for ${stem(r.recording.path)}`}
                          className="text cell"
                          style={{ width: 56 }}
                          value={r.run}
                          inputMode="numeric"
                          onChange={(e) => update(i, { run: e.target.value.replace(/\D/g, "") })}
                        />
                      </td>
                      <td className="small" style={{ color: "var(--warn-text)" }}>
                        {issue}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="small muted">
            The person's ID is kept only in the private subject key; in the dataset each person
            gets a neutral label (001, 002, …). Sessions, tasks and runs become part of the BIDS
            file names, so they may use only letters and digits.
          </p>
        </section>
      ) : null}

      {step === 3 ? (
        <section className="stack">
          {dataset ? null : (
            <div className="card">
              <h2>The new dataset</h2>
              <label className="field">
                Name
                <input
                  className="text mono"
                  value={name}
                  placeholder="e.g. oddball_study"
                  onChange={(e) => setName(e.target.value.trim())}
                />
              </label>
              {target ? <div className="small muted mono">{target}</div> : null}
            </div>
          )}
          <div className="card">
            <h2>Personal details that will be removed</h2>
            <ul style={{ margin: 0, paddingLeft: 18, lineHeight: 1.6 }}>
              <li>
                <b>Header fields:</b> names, IDs, experimenter, free-text descriptions, project
                name, device serial number and site, machine IDs and time zone.
              </li>
              <li>
                <b>Dates:</b> the recording and birth dates move back by a random shift for each
                person, so age at recording, times of day and the time between sessions are kept.
              </li>
              <li>
                <b>Event text:</b> the person's name, ID and birth date, dates, e-mail addresses and
                phone numbers become “[removed]”. Other free text is listed after the import for
                you to check.
              </li>
              <li>
                <b>File names</b> are not carried over: files are named after the neutral label.
              </li>
            </ul>
          </div>
          <div className="card">
            <h2>The subject key</h2>
            <p className="small muted">
              The only file that links the neutral labels to the people's IDs, and the only place
              the date shifts are kept. Keep it private, back it up, and never put it inside the
              dataset. Import more recordings of the same people with the same key.
            </p>
            <div className="row" style={{ gap: 8, flexWrap: "nowrap" }}>
              <input
                className="text mono"
                style={{ flex: 1 }}
                value={key}
                aria-label="Subject key file"
                onChange={(e) => setKeyPath(e.target.value)}
              />
              <button
                type="button"
                className="btn"
                onClick={() =>
                  void window.braino
                    .choose({
                      kind: "save",
                      title: "Subject key",
                      defaultPath: key,
                      filters: [{ name: "Subject key", extensions: ["json"] }],
                    })
                    .then(([p]) => p && setKeyPath(p))
                }
              >
                Choose…
              </button>
            </div>
          </div>
        </section>
      ) : null}

      {step === 4 ? (
        <section className="stack">
          {job ? (
            <div className="card compact">
              <JobLine job={job} />
            </div>
          ) : null}
          {results ? (
            <ImportResults rows={rows} results={results} />
          ) : (
            <ImportEvents jobId={jobId} rows={rows} />
          )}
          {job?.status === "done" ? (
            <div className="row">
              <span className="spacer" />
              <button type="button" className="btn primary big" onClick={() => void openDataset(target)}>
                Open the dataset
              </button>
            </div>
          ) : null}
        </section>
      ) : null}

      <ErrorNotice error={error} />

      {step < 4 ? (
        <div className="row">
          <button type="button" className="btn big" disabled={step === 0} onClick={() => setStep(step - 1)}>
            Back
          </button>
          <span className="spacer" />
          {step === 2 && problems.size > 0 ? (
            <span className="small" style={{ color: "var(--warn-text)" }}>
              {plural(problems.size, "row needs", "rows need")} attention
            </span>
          ) : null}
          {step < 3 ? (
            <button type="button" className="btn primary big" disabled={!canNext} onClick={() => setStep(step + 1)}>
              Next
            </button>
          ) : (
            <button type="button" className="btn primary big" disabled={!canNext || importing} onClick={() => void start()}>
              Import {plural(rows.length, "recording")}
            </button>
          )}
        </div>
      ) : null}
    </main>
  );
}

function ChooseStep({
  busy,
  onAdd,
  onChoose,
}: {
  busy: boolean;
  onAdd: (paths: string[]) => Promise<void>;
  onChoose: (kind: "files" | "folders") => Promise<void>;
}) {
  const [over, setOver] = useState(false);
  const drop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    const paths = [...e.dataTransfer.files].map((f) => window.braino.pathForFile(f)).filter(Boolean);
    void onAdd(paths);
  };
  return (
    <section
      className={`dropzone${over ? " over" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={drop}
    >
      <div className="stack" style={{ alignItems: "center" }}>
        <h2 style={{ color: "var(--text)" }}>{busy ? "Looking for recordings…" : "Drop recordings or folders here"}</h2>
        <p>
          EDF/BDF, BrainVision, FIF, EEGLAB, Neuroscan CNT, EGI MFF, GDF, Curry, Nihon Kohden and
          Persyst. Folders are searched for recordings.
        </p>
        <div className="row">
          <button type="button" className="btn" disabled={busy} onClick={() => void onChoose("files")}>
            Choose files…
          </button>
          <button type="button" className="btn" disabled={busy} onClick={() => void onChoose("folders")}>
            Choose a folder…
          </button>
        </div>
      </div>
    </section>
  );
}

function ImportEvents({ jobId, rows }: { jobId: string | null; rows: Row[] }) {
  const { jobs } = useApp();
  useJobs(); // re-render as events arrive
  const events = jobId ? (jobs.eventsOf(jobId) as unknown as ImportResultItem[]) : [];
  return <ImportResults rows={rows} results={events} />;
}

function ImportResults({ rows, results }: { rows: Row[]; results: ImportResultItem[] }) {
  if (results.length === 0) return null;
  const failed = results.filter((r) => !r.ok).length;
  const review = results.filter((r) => r.ok && r.check_by_hand && r.check_by_hand.length > 0);
  return (
    <div className="stack">
      {failed ? <div className="notice error">{plural(failed, "recording")} could not be imported.</div> : null}
      {review.length > 0 ? (
        <div className="notice warn">
          Some recordings have event text a machine cannot check for personal details. Look
          through it below, and remove anything that identifies a person.
        </div>
      ) : null}
      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr>
              <th>File</th>
              <th>Subject</th>
              <th>Result</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r) => (
              <tr key={r.index}>
                <td className="mono">{rows[r.index] ? stem(rows[r.index].recording.path) : r.index}</td>
                <td className="mono">{r.subject ? `sub-${r.subject}` : ""}</td>
                <td>
                  {r.ok ? (
                    <>
                      <Pill tone="ok">Imported</Pill>
                      {r.check_by_hand?.length ? (
                        <div className="small" style={{ marginTop: 4, color: "var(--warn-text)" }}>
                          Check by hand: {r.check_by_hand.join(" · ")}
                        </div>
                      ) : null}
                    </>
                  ) : (
                    <>
                      <Pill tone="danger">Failed</Pill>
                      <div className="small muted" style={{ marginTop: 4 }}>
                        {r.error}
                      </div>
                    </>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** What is wrong with each row, by index: a missing ID or task, or two rows that would be
 * the same recording in the dataset. */
export function rowProblems(rows: Pick<Row, "subject" | "session" | "task" | "run">[]): Map<number, string> {
  const problems = new Map<number, string>();
  const seen = new Map<string, number>();
  rows.forEach((r, i) => {
    if (!r.subject.trim()) problems.set(i, "Person's ID needed");
    else if (!r.task) problems.set(i, "Task needed");
    else {
      const id = [r.subject.trim(), r.session, r.task, r.run].join("|");
      const first = seen.get(id);
      if (first !== undefined) {
        problems.set(i, `Same person, session, task and run as row ${first + 1}: give a run or session`);
      } else {
        seen.set(id, i);
      }
    }
  });
  return problems;
}
