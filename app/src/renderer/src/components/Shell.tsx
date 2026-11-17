/** The window's frame: title bar, sidebar (with a dataset open), agent panel, status bar. */

import { type ReactNode, useEffect, useState } from "react";
import type { MemoryInfo } from "../../../shared/ipc";
import { basename, shortModelName } from "../lib/format";
import { type Route, useApp, useJobs } from "../state/app";
import { SUBJECT_STATUS, useDataset } from "../state/dataset";
import { isActive } from "../state/jobs";
import * as Icon from "./icons";
import { ProgressBar } from "./ui";

export function TitleBar({ crumbs }: { crumbs?: ReactNode }) {
  const { go, closeDataset, dataset, route } = useApp();
  return (
    <header className="titlebar">
      <button type="button" className="brand" onClick={closeDataset}>
        <Icon.Logo />
        BrainO
      </button>
      <nav className="crumbs" aria-label="Location">
        {route.name === "first-run" ? null : dataset ? (
          <>
            <button type="button" onClick={closeDataset}>
              Datasets
            </button>
            <span aria-hidden="true">/</span>
            <button type="button" className="current" onClick={() => go({ name: "overview" })}>
              {basename(dataset)}
            </button>
          </>
        ) : (
          <span className={route.name === "home" ? "current" : ""}>Datasets</span>
        )}
        {crumbs}
      </nav>
      <div className="spacer" />
      {route.name === "first-run" ? null : (
        <button
          type="button"
          className="icon-btn"
          aria-label="Settings"
          title="Settings"
          onClick={() => go({ name: "settings" })}
        >
          <Icon.Settings />
        </button>
      )}
    </header>
  );
}

const NAV: { route: Route["name"]; label: string }[] = [
  { route: "overview", label: "Overview" },
  { route: "import", label: "Import recordings" },
  { route: "pipelines", label: "Pipelines" },
  { route: "builder", label: "Pipeline builder" },
  { route: "decisions", label: "Decisions" },
  { route: "results", label: "Results" },
];

const SHOWN_SUBJECTS = 60;

export function Sidebar() {
  const { route, go } = useApp();
  const info = useDataset();
  const subjects = info.summary?.subjects ?? [];
  return (
    <aside className="sidebar">
      <nav className="nav" aria-label="Workspace">
        {NAV.map((item) => (
          <button
            key={item.route}
            type="button"
            className="nav-item"
            aria-current={route.name === item.route ? "page" : undefined}
            onClick={() => go({ name: item.route } as Route)}
          >
            <span>{item.label}</span>
            {item.route === "decisions" && info.pending.length > 0 ? (
              <span className="badge" aria-label={`${info.pending.length} waiting`}>
                {info.pending.length}
              </span>
            ) : null}
          </button>
        ))}
      </nav>
      <div>
        <div className="section-label">Subjects</div>
        {subjects.length === 0 ? <div className="tiny muted" style={{ padding: "0 10px" }}>None yet</div> : null}
        {subjects.slice(0, SHOWN_SUBJECTS).map((subject) => {
          const status = SUBJECT_STATUS[info.subjectStatus(subject)];
          return (
            <button
              key={subject}
              type="button"
              className="subject-item"
              aria-current={route.name === "subject" && route.subject === subject ? "page" : undefined}
              onClick={() => go({ name: "subject", subject })}
            >
              <span className="dot" style={{ background: status.color }} />
              <span className="mono">sub-{subject}</span>
              <span className="tiny muted" style={{ marginLeft: "auto" }}>
                {status.label}
              </span>
            </button>
          );
        })}
        {subjects.length > SHOWN_SUBJECTS ? (
          <div className="tiny muted" style={{ padding: "6px 10px" }}>
            and {subjects.length - SHOWN_SUBJECTS} more (see the Overview)
          </div>
        ) : null}
      </div>
    </aside>
  );
}

export function AgentPanel() {
  const { settings, updateSettings, system } = useApp();
  const open = settings?.agentPanelOpen ?? true;
  const model = system?.model.model;
  const full = system?.models.find((m) => m.id === model)?.name;
  const name = full ? shortModelName(full) : undefined;
  const toggle = () => void updateSettings({ agentPanelOpen: !open });
  if (!open) {
    return (
      <aside className="agent collapsed" aria-label="Agent">
        <div className="agent-head">
          <button type="button" className="icon-btn" aria-label="Open the agent panel" title="Agent" onClick={toggle}>
            <Icon.ChevronLeft />
          </button>
        </div>
      </aside>
    );
  }
  return (
    <aside className="agent" aria-label="Agent">
      <div className="agent-head">
        <span style={{ fontWeight: 600 }}>Agent</span>
        {name ? (
          <span className="pill ok ellipsis" title={full}>
            {name} · {system?.model.placement === "gpu" ? "GPU" : "CPU"} · on this computer
          </span>
        ) : (
          <span className="pill todo">No model set up</span>
        )}
        <span className="spacer" />
        <button type="button" className="icon-btn" aria-label="Collapse the agent panel" onClick={toggle}>
          <Icon.ChevronRight />
        </button>
      </div>
      <div className="agent-body">
        <p>
          The agent will work here: ask about your data in your own words, and it uses the same
          tools as the menus, asking you before anything that drops data.
        </p>
        <div className="notice">
          The agent panel is the next part of the app to be built. Until then, the agent runs
          in a terminal: <span className="mono">braino agent --dataset &lt;folder&gt;</span>.
        </div>
      </div>
    </aside>
  );
}

export function StatusBar() {
  const { engine, system } = useApp();
  const jobs = useJobs();
  const [memory, setMemory] = useState<MemoryInfo | null>(null);

  useEffect(() => {
    let alive = true;
    const update = () => void window.braino.memory().then((m) => alive && setMemory(m));
    update();
    const timer = window.setInterval(update, 5000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, []);

  const running = jobs.filter((j) => j.status === "running");
  const queued = jobs.filter((j) => j.status === "queued").length;
  const known = running.filter((j) => j.progress != null);
  const mean = known.length ? known.reduce((s, j) => s + (j.progress ?? 0), 0) / known.length : null;
  const model = system?.models.find((m) => m.id === system.model.model);
  const engineColor = {
    running: "var(--ok)",
    starting: "var(--accent)",
    stopped: "var(--muted-2)",
    failed: "var(--danger)",
  }[engine.state];
  const engineLabel = {
    running: "Engine running",
    starting: "Engine starting",
    stopped: "Engine stopped",
    failed: "Engine failed",
  }[engine.state];

  return (
    <footer className="statusbar">
      <span className="item">
        <span className="dot" style={{ background: engineColor }} />
        {engineLabel}
      </span>
      <span>
        Model:{" "}
        {model
          ? `${shortModelName(model.name)} · ${system?.model.placement === "gpu" ? "GPU" : "CPU"}`
          : "not set up"}
      </span>
      <span className="item">
        {running.length === 0
          ? "No jobs running"
          : `Jobs: ${running.length} running${queued ? `, ${queued} waiting` : ""}`}
        {running.length > 0 ? (
          <span style={{ width: 120, display: "inline-block" }}>
            <ProgressBar value={mean} thin label="Running jobs" />
          </span>
        ) : null}
      </span>
      {memory ? (
        <span>
          Memory: {((memory.totalBytes - memory.freeBytes) / 2 ** 30).toFixed(1)} of{" "}
          {Math.round(memory.totalBytes / 2 ** 30)} GB
        </span>
      ) : null}
      <span className="spacer" />
      <span>Nothing leaves this computer</span>
    </footer>
  );
}

/** Jobs not tied to a dataset (the model download, an example) for the screens that care. */
export function useGlobalJobs() {
  return useJobs().filter((j) => !j.dataset && isActive(j));
}
