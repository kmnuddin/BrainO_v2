/** Small charts drawn with SVG and canvas: no chart library, so they render offline, follow
 * the app's colours in light and dark, and stay fast. */

import { useEffect, useMemo, useRef } from "react";

export interface Layout {
  names: string[];
  x: number[];
  y: number[];
}

const PAD = 0.08;

/** Electrode positions (0-1, nose up) to SVG units inside a head of radius 0.5 at (0.5, 0.5). */
function place(layout: Layout) {
  const xs = layout.x;
  const ys = layout.y;
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const span = Math.max(maxX - minX, maxY - minY) || 1;
  const scale = (1 - 2 * PAD - 0.08) / span;
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  return layout.names.map((name, i) => ({
    name,
    x: 0.5 + (xs[i] - cx) * scale,
    // Layout y grows up (nose); SVG y grows down.
    y: 0.5 - (ys[i] - cy) * scale,
  }));
}

/** A diverging colour for -1..1 (blue, white, red) that reads in both themes. */
export function diverging(t: number): [number, number, number] {
  const v = Math.max(-1, Math.min(1, t));
  const white = [245, 245, 245];
  const end = v < 0 ? [49, 104, 196] : [204, 64, 52];
  const a = Math.abs(v);
  return [0, 1, 2].map((k) => Math.round(white[k] + (end[k] - white[k]) * a)) as [number, number, number];
}

/** A head seen from above with its electrodes. ``values`` (one per electrode) colour the head
 * as a map; ``marked`` electrodes are ringed, ``labelled`` ones named. */
export function HeadMap({
  layout,
  values,
  marked = [],
  secondary = [],
  labelled,
  size = 220,
  title,
}: {
  layout: Layout;
  values?: number[];
  marked?: string[];
  secondary?: string[];
  labelled?: string[];
  size?: number;
  title?: string;
}) {
  const points = useMemo(() => place(layout), [layout]);
  const canvas = useRef<HTMLCanvasElement>(null);

  // Inverse-distance-weighted map of the values inside the head.
  useEffect(() => {
    const el = canvas.current;
    if (!el || !values) return;
    const n = 64;
    el.width = n;
    el.height = n;
    const ctx = el.getContext("2d");
    if (!ctx) return;
    const image = ctx.createImageData(n, n);
    const max = Math.max(...values.map(Math.abs)) || 1;
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const x = (i + 0.5) / n;
        const y = (j + 0.5) / n;
        const o = (j * n + i) * 4;
        if ((x - 0.5) ** 2 + (y - 0.5) ** 2 > 0.25) {
          image.data[o + 3] = 0;
          continue;
        }
        let num = 0;
        let den = 0;
        for (let k = 0; k < points.length; k++) {
          const d2 = (points[k].x - x) ** 2 + (points[k].y - y) ** 2 + 1e-6;
          const w = 1 / (d2 * d2);
          num += w * values[k];
          den += w;
        }
        const [r, g, b] = diverging(num / den / max);
        image.data[o] = r;
        image.data[o + 1] = g;
        image.data[o + 2] = b;
        image.data[o + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);
  }, [values, points]);

  const ring = new Set(marked);
  const ring2 = new Set(secondary);
  const named = new Set(labelled ?? [...marked, ...secondary]);
  return (
    <figure style={{ margin: 0, width: size }} aria-label={title ?? "Electrode map"}>
      <div style={{ position: "relative", width: size, height: size }}>
        {values ? (
          <canvas
            ref={canvas}
            style={{ position: "absolute", inset: 0, width: size, height: size, imageRendering: "auto", borderRadius: "50%" }}
          />
        ) : null}
        <svg viewBox="0 0 1 1" width={size} height={size} style={{ position: "absolute", inset: 0 }}>
          <circle cx={0.5} cy={0.5} r={0.5 - 0.01} fill="none" stroke="var(--muted)" strokeWidth={0.008} />
          <path d="M0.46 0.015 L0.5 -0.03 L0.54 0.015" fill="none" stroke="var(--muted)" strokeWidth={0.008} transform="translate(0 0.02)" />
          {points.map((p) => (
            <g key={p.name}>
              <circle
                cx={p.x}
                cy={p.y}
                r={ring.has(p.name) || ring2.has(p.name) ? 0.022 : 0.011}
                fill={ring.has(p.name) ? "var(--warn)" : ring2.has(p.name) ? "var(--ok)" : values ? "rgba(0,0,0,0.55)" : "var(--muted-2)"}
                stroke={ring.has(p.name) || ring2.has(p.name) ? "var(--surface)" : "none"}
                strokeWidth={0.006}
              >
                <title>{p.name}</title>
              </circle>
              {named.has(p.name) ? (
                <text x={p.x} y={p.y - 0.032} textAnchor="middle" fontSize={0.045} fill="var(--text)" fontWeight={600}>
                  {p.name}
                </text>
              ) : null}
            </g>
          ))}
        </svg>
      </div>
      {title ? <figcaption className="tiny muted" style={{ textAlign: "center" }}>{title}</figcaption> : null}
    </figure>
  );
}

export interface Series {
  label: string;
  values: number[];
  color: string;
  dashed?: boolean;
}

/** Lines over a shared x axis, with ticks and an optional shaded span. */
export function LineChart({
  x,
  series,
  xLabel,
  yLabel,
  height = 200,
  span,
  marks = [],
  invertY = false,
  width = 600,
}: {
  x: number[];
  series: Series[];
  xLabel: string;
  yLabel: string;
  height?: number;
  span?: [number, number] | null;
  /** Shaded bands along x (e.g. significant clusters) drawn under the axis. */
  marks?: [number, number][];
  invertY?: boolean;
  /** Drawing width: smaller for small charts, so their labels stay readable. */
  width?: number;
}) {
  const m = { l: 52, r: 12, t: 10, b: 34 };
  const all = series.flatMap((s) => s.values).filter(Number.isFinite);
  if (x.length < 2 || all.length === 0) return <div className="empty">No data</div>;
  let lo = Math.min(...all);
  let hi = Math.max(...all);
  if (lo === hi) {
    lo -= 1;
    hi += 1;
  }
  const pad = (hi - lo) * 0.05;
  lo -= pad;
  hi += pad;
  const x0 = x[0];
  const x1 = x[x.length - 1];
  const px = (v: number) => m.l + ((v - x0) / (x1 - x0)) * (width - m.l - m.r);
  const py = (v: number) => {
    const t = (v - lo) / (hi - lo);
    return m.t + (invertY ? t : 1 - t) * (height - m.t - m.b);
  };
  const xt = ticks(x0, x1, 6);
  const yt = ticks(lo, hi, 4);
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" style={{ maxWidth: width, display: "block" }} role="img" aria-label={`${yLabel} by ${xLabel}`}>
      {span ? (
        <rect x={px(span[0])} y={m.t} width={Math.max(0, px(span[1]) - px(span[0]))} height={height - m.t - m.b} fill="var(--accent-soft)" />
      ) : null}
      {marks.map(([a, b], i) => (
        <rect key={i} x={px(a)} y={height - m.b - 5} width={Math.max(2, px(b) - px(a))} height={5} fill="var(--warn)" />
      ))}
      {yt.map((t) => (
        <g key={`y${t}`}>
          <line x1={m.l} x2={width - m.r} y1={py(t)} y2={py(t)} stroke="var(--border-soft)" />
          <text x={m.l - 6} y={py(t) + 4} textAnchor="end" fontSize={11} fill="var(--muted)">
            {fmt(t)}
          </text>
        </g>
      ))}
      {lo < 0 && hi > 0 ? <line x1={m.l} x2={width - m.r} y1={py(0)} y2={py(0)} stroke="var(--muted-2)" /> : null}
      {x0 < 0 && x1 > 0 ? <line x1={px(0)} x2={px(0)} y1={m.t} y2={height - m.b} stroke="var(--muted-2)" strokeDasharray="3 3" /> : null}
      {xt.map((t) => (
        <text key={`x${t}`} x={px(t)} y={height - m.b + 16} textAnchor="middle" fontSize={11} fill="var(--muted)">
          {fmt(t)}
        </text>
      ))}
      <text x={(m.l + width - m.r) / 2} y={height - 4} textAnchor="middle" fontSize={11} fill="var(--muted)">
        {xLabel}
      </text>
      <text x={12} y={(m.t + height - m.b) / 2} textAnchor="middle" fontSize={11} fill="var(--muted)" transform={`rotate(-90 12 ${(m.t + height - m.b) / 2})`}>
        {yLabel}
      </text>
      {series.map((s) => (
        <polyline
          key={s.label}
          fill="none"
          stroke={s.color}
          strokeWidth={1.6}
          strokeDasharray={s.dashed ? "5 4" : undefined}
          points={s.values.map((v, i) => `${px(x[i]).toFixed(1)},${py(v).toFixed(1)}`).join(" ")}
        />
      ))}
    </svg>
  );
}

export function Legend({ series }: { series: Pick<Series, "label" | "color" | "dashed">[] }) {
  return (
    <div className="row tiny" style={{ gap: 14 }}>
      {series.map((s) => (
        <span key={s.label} className="row" style={{ gap: 6 }}>
          <svg width={18} height={8} aria-hidden="true">
            <line x1={0} x2={18} y1={4} y2={4} stroke={s.color} strokeWidth={2} strokeDasharray={s.dashed ? "4 3" : undefined} />
          </svg>
          {s.label}
        </span>
      ))}
    </div>
  );
}

/** A heat map (e.g. time-frequency power) on a canvas, values coloured on a diverging scale. */
export function HeatMap({
  values,
  x,
  y,
  xLabel,
  yLabel,
  height = 220,
}: {
  values: number[][];
  x: number[];
  y: number[];
  xLabel: string;
  yLabel: string;
  height?: number;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const el = canvas.current;
    if (!el || values.length === 0) return;
    const rows = values.length;
    const cols = values[0].length;
    el.width = cols;
    el.height = rows;
    const ctx = el.getContext("2d");
    if (!ctx) return;
    const max = Math.max(...values.flat().map(Math.abs)) || 1;
    const image = ctx.createImageData(cols, rows);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const [R, G, B] = diverging(values[rows - 1 - r][c] / max);
        const o = (r * cols + c) * 4;
        image.data[o] = R;
        image.data[o + 1] = G;
        image.data[o + 2] = B;
        image.data[o + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);
  }, [values]);
  return (
    <div className="stack" style={{ gap: 4 }}>
      <div className="row" style={{ gap: 8, flexWrap: "nowrap", alignItems: "stretch" }}>
        <div className="tiny muted" style={{ writingMode: "vertical-rl", transform: "rotate(180deg)", textAlign: "center" }}>
          {yLabel} ({fmt(y[0])}–{fmt(y[y.length - 1])})
        </div>
        <canvas ref={canvas} style={{ width: "100%", maxWidth: 600, height, imageRendering: "pixelated", borderRadius: 4 }} />
      </div>
      <div className="tiny muted" style={{ textAlign: "center" }}>
        {xLabel} ({fmt(x[0])} to {fmt(x[x.length - 1])})
      </div>
    </div>
  );
}

function ticks(lo: number, hi: number, count: number): number[] {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((f) => f * power).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Number(v.toFixed(10)));
  return out;
}

function fmt(v: number): string {
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 1) return String(Number(v.toFixed(1)));
  return String(Number(v.toFixed(2)));
}
