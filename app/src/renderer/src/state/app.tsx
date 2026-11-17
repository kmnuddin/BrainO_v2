/** The app's state: the engine, settings, the open dataset and the screen shown. */

import {
  type ReactNode,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  useSyncExternalStore,
} from "react";
import type { AppSettings, EngineStatus } from "../../../shared/ipc";
import { api } from "../engine/api";
import type { Job, PipelineSpec, SystemInfo, ToolInfo } from "../engine/types";
import { JobStore, isActive, samePath } from "./jobs";

export type Route =
  | { name: "first-run" }
  | { name: "home" }
  | { name: "settings" }
  | { name: "overview" }
  | { name: "import"; newDataset?: boolean }
  | { name: "pipelines"; pipeline?: string }
  | { name: "builder"; pipeline?: string; draft?: PipelineSpec }
  | { name: "decisions"; proposal?: string }
  | { name: "results" }
  | { name: "subject"; subject: string };

interface AppState {
  engine: EngineStatus;
  system: SystemInfo | null;
  settings: AppSettings | null;
  jobs: JobStore;
  route: Route;
  /** The open dataset's folder. */
  dataset: string | null;
  /** Bumped when the dataset's files change (a job on it finished), to reload views. */
  datasetVersion: number;
  go(route: Route): void;
  openDataset(path: string, route?: Route): Promise<void>;
  closeDataset(): void;
  updateSettings(changes: Partial<AppSettings>): Promise<void>;
  setSettings(settings: AppSettings): void;
  refreshSystem(): Promise<void>;
}

const Context = createContext<AppState | null>(null);

export function useApp(): AppState {
  const state = useContext(Context);
  if (!state) throw new Error("useApp outside AppProvider");
  return state;
}

export function useJobs(): Job[] {
  const { jobs } = useApp();
  return useSyncExternalStore(jobs.subscribe, jobs.list);
}

/** The open dataset's unfinished jobs. */
export function useDatasetJobs(): Job[] {
  const { dataset } = useApp();
  const jobs = useJobs();
  return useMemo(() => jobs.filter((j) => samePath(j.dataset, dataset)), [jobs, dataset]);
}

let toolsCache: Promise<ToolInfo[]> | null = null;

/** The engine's tools, read once per engine start. */
export function useTools(): Map<string, ToolInfo> | undefined {
  const { engine } = useApp();
  const [tools, setTools] = useState<Map<string, ToolInfo>>();
  useEffect(() => {
    if (engine.state !== "running") {
      toolsCache = null;
      return;
    }
    let alive = true;
    toolsCache ??= api.tools().catch((e: unknown) => {
      toolsCache = null;
      throw e;
    });
    void toolsCache.then((list) => alive && setTools(new Map(list.map((t) => [t.name, t]))));
    return () => {
      alive = false;
    };
  }, [engine.state]);
  return tools;
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [engine, setEngine] = useState<EngineStatus>({ state: "starting", log: [] });
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [route, setRoute] = useState<Route>({ name: "home" });
  const [dataset, setDataset] = useState<string | null>(null);
  const [datasetVersion, setDatasetVersion] = useState(0);
  const [jobs] = useState(() => new JobStore());

  useEffect(() => {
    let alive = true;
    void window.braino.settings().then((s) => {
      if (!alive) return;
      setSettings(s);
      if (!s.firstRunDone) setRoute({ name: "first-run" });
    });
    void window.braino.engineStatus().then((s) => alive && setEngine(s));
    const offStatus = window.braino.onEngineStatus((s) => {
      setEngine(s);
      if (s.state !== "running") jobs.reset();
    });
    const offEvent = window.braino.onEvent((m) => jobs.apply(m));
    return () => {
      alive = false;
      offStatus();
      offEvent();
    };
  }, [jobs]);

  const refreshSystem = useCallback(async () => {
    setSystem(await api.system());
  }, []);

  useEffect(() => {
    if (engine.state === "running") void refreshSystem().catch(() => undefined);
  }, [engine.state, refreshSystem]);

  // A finished job that wrote into the open dataset (or set up the model) reloads its views.
  useEffect(() => {
    const seen = new Map<string, string>();
    return jobs.subscribe(() => {
      for (const job of jobs.list()) {
        const before = seen.get(job.id);
        seen.set(job.id, job.status);
        if (before === undefined || before === job.status || isActive(job)) continue;
        if (job.kind === "model_setup") void refreshSystem().catch(() => undefined);
        if (job.dataset) setDatasetVersion((v) => v + 1);
      }
    });
  }, [jobs, refreshSystem]);

  const openDataset = useCallback(async (path: string, next: Route = { name: "overview" }) => {
    setSettings(await window.braino.openedDataset(path));
    setDataset(path);
    setRoute(next);
  }, []);

  const value = useMemo<AppState>(
    () => ({
      engine,
      system,
      settings,
      jobs,
      route,
      dataset,
      datasetVersion,
      go: setRoute,
      openDataset,
      closeDataset: () => {
        setDataset(null);
        setRoute({ name: "home" });
      },
      updateSettings: async (changes) => {
        setSettings(await window.braino.updateSettings(changes));
      },
      setSettings,
      refreshSystem,
    }),
    [engine, system, settings, jobs, route, dataset, datasetVersion, openDataset, refreshSystem],
  );
  return <Context.Provider value={value}>{children}</Context.Provider>;
}
