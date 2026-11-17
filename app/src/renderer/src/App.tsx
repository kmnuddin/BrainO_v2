import type { ReactNode } from "react";
import { AgentPanel, Sidebar, StatusBar, TitleBar } from "./components/Shell";
import { Loading } from "./components/ui";
import { Builder } from "./screens/Builder";
import { Decisions } from "./screens/Decisions";
import { FirstRun } from "./screens/FirstRun";
import { Pipelines } from "./screens/Pipelines";
import { Results } from "./screens/Results";
import { Home } from "./screens/Home";
import { ImportWizard } from "./screens/ImportWizard";
import { Overview } from "./screens/Overview";
import { Settings } from "./screens/Settings";
import { Subject } from "./screens/Subject";
import { useApp } from "./state/app";
import { DatasetProvider } from "./state/dataset";

// Each visit to a screen (each go()) starts it afresh, with what the route carries.
const routeKeys = new WeakMap<object, number>();
let nextKey = 0;
function keyOf(route: object): number {
  let key = routeKeys.get(route);
  if (key === undefined) {
    key = nextKey++;
    routeKeys.set(route, key);
  }
  return key;
}

export function App() {
  const { route, dataset, engine, settings } = useApp();

  if (!settings) return <Loading what="Starting" />;
  if (route.name === "first-run") {
    return (
      <div className="app">
        <TitleBar />
        <div className="body">{engine.state === "running" ? <FirstRun /> : <EngineState />}</div>
      </div>
    );
  }

  let screen: ReactNode;
  if (route.name === "settings") screen = <Settings />;
  else if (engine.state !== "running") screen = <EngineState />;
  else if (!dataset) screen = route.name === "import" ? <ImportWizard /> : <Home />;
  else {
    screen = {
      home: <Home />,
      overview: <Overview />,
      import: <ImportWizard />,
      pipelines: route.name === "pipelines" ? <Pipelines key={keyOf(route)} initial={route.pipeline} /> : null,
      builder:
        route.name === "builder" ? <Builder key={keyOf(route)} pipeline={route.pipeline} draft={route.draft} /> : null,
      decisions: route.name === "decisions" ? <Decisions key={keyOf(route)} initial={route.proposal} /> : null,
      results: <Results />,
      subject: route.name === "subject" ? <Subject subject={route.subject} /> : null,
      "first-run": null,
      settings: null,
    }[route.name];
  }

  const body = (
    <div className="body">
      {dataset && engine.state === "running" ? <Sidebar /> : null}
      {screen}
      <AgentPanel />
    </div>
  );
  return (
    <div className="app">
      <TitleBar />
      {dataset && engine.state === "running" ? (
        <DatasetProvider key={dataset} path={dataset}>
          {body}
        </DatasetProvider>
      ) : (
        body
      )}
      <StatusBar />
    </div>
  );
}

function EngineState() {
  const { engine, go } = useApp();
  if (engine.state === "starting") return <Loading what="Starting the engine" />;
  return (
    <main className="workspace narrow">
      <h1>The engine is not running</h1>
      {engine.error ? <div className="notice error">{engine.error}</div> : null}
      {engine.log.length ? <pre className="log">{engine.log.join("\n")}</pre> : null}
      <div className="row">
        <button type="button" className="btn primary" onClick={() => void window.braino.restartEngine()}>
          Start it again
        </button>
        <button type="button" className="btn" onClick={() => go({ name: "settings" })}>
          Choose its Python…
        </button>
      </div>
      <p className="small muted">
        BrainO's engine is a Python program. If it is not installed, install it with{" "}
        <span className="mono">pip install "braino[eeg,server]"</span>, then choose its Python in Settings.
      </p>
    </main>
  );
}
