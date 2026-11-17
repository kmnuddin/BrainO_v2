/** Results: ERPs with the measurement window and significant clusters, scalp maps, spectra and
 * time-frequency maps, for the group or one subject; the measures and the group test as
 * tables; figures and data exported. */

import { useEffect, useMemo, useRef, useState } from "react";
import { HeadMap, HeatMap, Legend, LineChart, type Series } from "../components/charts";
import { ErrorNotice, Loading, Pill } from "../components/ui";
import { api } from "../engine/api";
import type { GroupResult, ResultData, ResultStep, ResultsIndex } from "../engine/types";
import { plural } from "../lib/format";
import { useAsync } from "../lib/useAsync";
import { useApp } from "../state/app";
import { useDataset } from "../state/dataset";

const COLORS = ["var(--accent)", "var(--warn)", "var(--ok)", "#9b59b6", "#16a2b8", "#d64575"];

export function Results() {
  const { datasetVersion, go } = useApp();
  const info = useDataset();
  const withResults = info.pipelines.filter(({ state }) => state && Object.keys(state.subjects).length > 0);
  const [name, setName] = useState<string | null>(null);
  const chosen = name ?? withResults[0]?.item.name ?? null;
  const index = useAsync(
    () => (chosen ? api.results(chosen, info.path) : Promise.resolve(null)),
    [chosen, info.path, datasetVersion],
  );
  const [stepId, setStepId] = useState<string | null>(null);
  const steps = index.data?.steps.filter((s) => s.subjects.length > 0) ?? [];
  const step = steps.find((s) => s.id === stepId) ?? steps[0] ?? null;

  if (withResults.length === 0) {
    return (
      <main className="workspace">
        <h1>Results</h1>
        <div className="card">
          <p className="muted">No pipeline has run on this dataset yet.</p>
          <div>
            <button type="button" className="btn primary" onClick={() => go({ name: "pipelines" })}>
              Run a pipeline
            </button>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="workspace">
      <div className="row">
        <h1>Results</h1>
        <label className="row small" style={{ gap: 8 }}>
          <span className="muted">Pipeline</span>
          <select className="text" value={chosen ?? ""} onChange={(e) => { setName(e.target.value); setStepId(null); }}>
            {withResults.map((p) => (
              <option key={p.item.name} value={p.item.name}>
                {p.item.name}
              </option>
            ))}
          </select>
        </label>
        <span className="spacer" />
        {index.data && index.data.measures.length ? <ExportMeasures index={index.data} /> : null}
      </div>
      <ErrorNotice error={index.error} />
      {!index.data && !index.error ? <Loading what="Reading the results" /> : null}
      {index.data ? (
        <>
          {steps.length > 1 ? (
            <div className="chips" role="group" aria-label="Measure">
              {steps.map((s) => (
                <button key={s.id} type="button" className="chip" aria-pressed={s.id === step?.id} onClick={() => setStepId(s.id)}>
                  {{ erp: "ERPs", psd: "Spectra", tfr: "Time–frequency" }[s.kind]} · <span className="mono">{s.id}</span>
                </button>
              ))}
            </div>
          ) : null}
          {step ? (
            <StepResults key={`${chosen}-${step.id}`} pipeline={chosen!} step={step} group={index.data.group} />
          ) : (
            <div className="card muted">
              This pipeline has no ERP, spectrum or time–frequency step with results yet.
            </div>
          )}
          {index.data.group.map((g) => (
            <GroupTable key={g.id} group={g} />
          ))}
          {index.data.measures.length ? <MeasuresTable index={index.data} /> : null}
        </>
      ) : null}
    </main>
  );
}

function StepResults({ pipeline, step, group }: { pipeline: string; step: ResultStep; group: GroupResult[] }) {
  const info = useDataset();
  const [subject, setSubject] = useState<string | null>(null);
  const [channel, setChannel] = useState<string | null>(step.windows[0]?.channels?.[0] ?? null);
  const [condition, setCondition] = useState<string | null>(null);
  const window = step.windows[0];
  const data = useAsync(
    () =>
      api.resultData(pipeline, step.id, {
        dataset: info.path,
        subject,
        channel,
        tmin: window?.tmin ?? null,
        tmax: window?.tmax ?? null,
        condition,
      }),
    [pipeline, step.id, info.path, subject, channel, condition],
  );
  const figure = useRef<HTMLDivElement>(null);
  const d = data.data;

  return (
    <section className="card">
      <div className="row">
        <h2>{{ erp: "ERPs", psd: "Power spectra", tfr: "Time–frequency power" }[step.kind]}</h2>
        <span className="small muted">
          {subject ? `sub-${subject}` : `mean of ${plural(step.subjects.length, "subject")}`}
        </span>
        <span className="spacer" />
        <label className="row small" style={{ gap: 6 }}>
          <span className="muted">Show</span>
          <select className="text" value={subject ?? ""} onChange={(e) => setSubject(e.target.value || null)}>
            <option value="">Group mean</option>
            {step.subjects.map((s) => (
              <option key={s} value={s}>
                sub-{s}
              </option>
            ))}
          </select>
        </label>
        {d ? (
          <label className="row small" style={{ gap: 6 }}>
            <span className="muted">Channel</span>
            <select className="text" value={d.channel} onChange={(e) => setChannel(e.target.value)}>
              {d.channels.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
        ) : null}
        {d?.kind === "tfr" ? (
          <label className="row small" style={{ gap: 6 }}>
            <span className="muted">Condition</span>
            <select className="text" value={d.name} onChange={(e) => setCondition(e.target.value)}>
              {d.names.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
        ) : null}
        {d && d.kind !== "tfr" ? (
          <button type="button" className="btn small" onClick={() => void exportFigure(figure.current, `${pipeline}-${step.id}-${d.channel}`)}>
            Save figure
          </button>
        ) : null}
        {d ? (
          <button type="button" className="btn small" onClick={() => void exportData(d, `${pipeline}-${step.id}-${d.channel}`)}>
            Save data
          </button>
        ) : null}
      </div>
      <ErrorNotice error={data.error} />
      {!d && !data.error ? <Loading what="Reading" /> : null}
      {d ? <div ref={figure}><Plot data={d} step={step} group={group} /></div> : null}
    </section>
  );
}

function Plot({ data, step, group }: { data: ResultData; step: ResultStep; group: GroupResult[] }) {
  // Significant clusters of the group test that include this channel, along time.
  const marks = useMemo(() => {
    if (data.kind !== "erp") return [];
    const out: [number, number][] = [];
    for (const g of group) {
      const clusters = g.result?.result?.clusters ?? [];
      const alpha = Number(g.result?.result?.alpha ?? 0.05);
      for (const c of clusters) {
        const time = c.ranges.time ?? c.ranges.times;
        if (c.p < alpha && time && c.channels.includes(data.channel)) out.push(time);
      }
    }
    return out;
  }, [group, data]);

  if (data.kind === "tfr") {
    return (
      <div className="stack">
        <HeatMap
          values={data.power}
          x={data.times}
          y={data.freqs}
          xLabel="time (s)"
          yLabel="Hz"
        />
        <div className="tiny muted">
          {data.name} at {data.channel}; colour: change in power ({data.unit}), red up, blue down.
        </div>
      </div>
    );
  }
  const names = Object.keys(data.series);
  const series: Series[] = names.map((n, i) => ({
    label: n,
    values: data.series[n],
    color: COLORS[i % COLORS.length],
    dashed: n.includes("_minus_"),
  }));
  const window = step.windows[0];

  return (
    <div className="stack">
      <LineChart
        x={data.x}
        series={series}
        xLabel={data.kind === "erp" ? "time (s)" : "frequency (Hz)"}
        yLabel={data.unit}
        height={260}
        span={data.kind === "erp" && window?.tmin != null && window.tmax != null ? [window.tmin, window.tmax] : null}
        marks={marks}
      />
      <div className="row">
        <Legend series={series} />
        <span className="spacer" />
        <span className="tiny muted">
          {data.kind === "erp" && window?.tmin != null ? `Shaded: the measured window (${window.tmin}–${window.tmax} s). ` : ""}
          {marks.length ? "Orange bars: significant clusters at this channel." : ""}
        </span>
      </div>
      {data.kind === "erp" && data.layout && data.maps && Object.keys(data.maps).length ? (
        <div className="stack" style={{ gap: 6 }}>
          <div className="small muted">
            Scalp maps: mean amplitude in {data.window?.[0]}–{data.window?.[1]} s (red positive, blue negative; each map
            scaled to its own range)
          </div>
          <div className="row" style={{ gap: 16, alignItems: "flex-start" }}>
            {Object.entries(data.maps).map(([n, values]) => (
              <HeadMap
                key={n}
                layout={data.layout!}
                values={data.layout!.names.map((ch) => values[data.channels.indexOf(ch)] ?? 0)}
                labelled={[data.channel]}
                size={150}
                title={`${n} (max ${Math.max(...values.map(Math.abs)).toFixed(1)} µV)`}
              />
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function GroupTable({ group }: { group: GroupResult }) {
  const result = group.result?.result;
  if (!result) {
    return (
      <section className="card compact">
        <h2>
          Group test · <span className="mono">{group.id}</span>
        </h2>
        <p className="small muted">Not run yet: it runs once every subject's steps are done.</p>
      </section>
    );
  }
  const clusters = result.clusters;
  return (
    <section className="card">
      <div className="row">
        <h2>
          Group test · <span className="mono">{group.id}</span>
        </h2>
        <span className="small muted">{String(result.test ?? group.tool)}</span>
        <span className="spacer" />
        <Pill tone={Number(result.n_significant ?? 0) > 0 ? "ok" : "todo"}>
          {Number(result.n_significant ?? 0) > 0 ? `${result.n_significant} significant` : "Nothing significant"}
        </Pill>
      </div>
      <div className="small muted">
        {plural(Number(result.n_subjects ?? 0), "subject")}
        {result.correction ? ` · correction ${String(result.correction)}` : ""}
        {result.alpha != null ? ` · α = ${String(result.alpha)}` : ""}
        {result.n_permutations ? ` · ${String(result.n_permutations)} permutations` : ""}
      </div>
      {clusters ? (
        clusters.length ? (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Cluster</th>
                  <th>p</th>
                  <th>Sign</th>
                  <th>Extent</th>
                  <th>Channels</th>
                </tr>
              </thead>
              <tbody>
                {clusters.map((c, i) => (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td>
                      <b>{c.p < 0.001 ? "< 0.001" : c.p.toFixed(3)}</b>
                    </td>
                    <td>{c.sign}</td>
                    <td className="small">
                      {Object.entries(c.ranges)
                        .map(([axis, [a, b]]) => `${axis} ${a}–${b}`)
                        .join(", ")}
                    </td>
                    <td className="small mono">{c.channels.join(", ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="small muted">No clusters formed.</p>
        )
      ) : (
        <div className="grid cols-4">
          {["min_p", "peak_stat", "peak_channel", "n_tests"].map((k) =>
            result[k] != null ? (
              <div key={k} className="tile">
                <div className="label">{{ min_p: "Smallest corrected p", peak_stat: "Largest statistic", peak_channel: "At channel", n_tests: "Points tested" }[k]}</div>
                <div className="value">{statValue(result[k])}</div>
              </div>
            ) : null,
          )}
        </div>
      )}
      {(group.result?.notes ?? []).map((n) => (
        <div key={n} className="small muted">
          {n}
        </div>
      ))}
    </section>
  );
}

function statValue(value: unknown): string {
  if (typeof value !== "number") return String(value);
  if (Number.isInteger(value)) return value.toLocaleString();
  return Math.abs(value) < 0.001 ? value.toExponential(1) : String(Number(value.toPrecision(3)));
}

function MeasuresTable({ index }: { index: ResultsIndex }) {
  const rows = index.measures;
  return (
    <section className="card">
      <h2>Measures</h2>
      <div className="table-wrap" style={{ maxHeight: 360 }}>
        <table className="table">
          <thead>
            <tr>
              <th>Subject</th>
              <th>Measure</th>
              <th>Waveform</th>
              <th>Channels</th>
              <th>Value (µV)</th>
              <th>Latency (ms)</th>
              <th>Trials</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                <td className="mono">sub-{r.subject}</td>
                <td>{r.measure}</td>
                <td>{r.waveform}</td>
                <td className="small mono">{r.channels.join(", ")}</td>
                <td>{r.value_uv.toFixed(2)}</td>
                <td>{r.latency_ms ?? ""}</td>
                <td>{r.n_trials ?? ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function ExportMeasures({ index }: { index: ResultsIndex }) {
  const [saved, setSaved] = useState<string | null>(null);
  useEffect(() => setSaved(null), [index]);
  const save = async () => {
    const header = "subject,step,measure,waveform,channels,value_uv,latency_ms,n_trials";
    const lines = index.measures.map((r) =>
      [r.subject, r.step, r.measure, r.waveform, `"${r.channels.join(" ")}"`, r.value_uv, r.latency_ms ?? "", r.n_trials ?? ""].join(","),
    );
    const path = await window.braino.saveTextFile({
      title: "Save the measures",
      defaultName: `${index.name}-measures.csv`,
      text: [header, ...lines].join("\n") + "\n",
      extension: "csv",
    });
    setSaved(path);
  };
  return (
    <span className="row small" style={{ gap: 8 }}>
      {saved ? <span className="muted">Saved</span> : null}
      <button type="button" className="btn small" onClick={() => void save()}>
        Save the measures (CSV)
      </button>
    </span>
  );
}

async function exportFigure(container: HTMLElement | null, name: string) {
  const svg = container?.querySelector("svg[role=img]");
  if (!svg) return;
  // Resolve the theme's colours so the file looks the same outside the app.
  const clone = svg.cloneNode(true) as SVGElement;
  const style = getComputedStyle(document.documentElement);
  const text = clone.outerHTML.replace(/var\((--[a-z0-9-]+)\)/g, (_, v: string) => style.getPropertyValue(v).trim() || "#888");
  await window.braino.saveTextFile({
    title: "Save the figure",
    defaultName: `${name}.svg`,
    text: `<?xml version="1.0" encoding="UTF-8"?>\n${text.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"')}`,
    extension: "svg",
  });
}

async function exportData(data: ResultData, name: string) {
  let csv: string;
  if (data.kind === "tfr") {
    csv = ["freq_hz," + data.times.join(","), ...data.power.map((row, i) => `${data.freqs[i]},${row.join(",")}`)].join("\n");
  } else {
    const names = Object.keys(data.series);
    csv = [`${data.kind === "erp" ? "time_s" : "freq_hz"},${names.join(",")}`, ...data.x.map((x, i) => [x, ...names.map((n) => data.series[n][i])].join(","))].join("\n");
  }
  await window.braino.saveTextFile({ title: "Save the data", defaultName: `${name}.csv`, text: csv + "\n", extension: "csv" });
}
