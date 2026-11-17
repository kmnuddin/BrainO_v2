/** Development only: drive the window from a script and save screenshots, to check screens
 * without clicking through them by hand. Runs when BRAINO_CAPTURE names a JSON file of steps:
 *
 *   [{"wait": 3000}, {"click": "Open a dataset"}, {"type": ["#datadir", "C:\\data"]},
 *    {"until": "Datasets"}, {"shot": "home.png"}, {"run": "document.title"}]
 *
 * ``click`` presses the first visible button or link whose text contains the words; ``until``
 * waits (up to 60 s) for the words to appear on the page. Screenshots go next to the script.
 * The app quits after the last step. Never used in a packaged app. */

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import type { BrowserWindow } from "electron";

/** Answers for the next file dialogs, set by ``choose`` steps (the dialog is not shown). */
export const queuedChoices: string[][] = [];

type CaptureStep =
  | { wait: number }
  | { choose: string[] }
  | { click: string }
  | { type: [string, string] }
  | { until: string; timeout?: number }
  | { shot: string }
  | { run: string };

const sleep = (ms: number) => new Promise((done) => setTimeout(done, ms));

export async function runCapture(window: BrowserWindow, scriptPath: string): Promise<void> {
  const steps = JSON.parse(readFileSync(scriptPath, "utf-8")) as CaptureStep[];
  const folder = dirname(scriptPath);
  const log: string[] = [`main: WebSocket ${typeof WebSocket}, Node ${process.versions.node}`];
  const js = (code: string) => window.webContents.executeJavaScript(code, true);
  for (const step of steps) {
    try {
      if ("wait" in step) {
        await sleep(step.wait);
      } else if ("choose" in step) {
        queuedChoices.push(step.choose);
      } else if ("click" in step) {
        const found = await js(`(() => {
          const want = ${JSON.stringify(step.click)}.toLowerCase();
          const items = [...document.querySelectorAll("button, a, [role=button], summary")];
          const text = (el) => (el.textContent + " " + (el.getAttribute("aria-label") || ""))
            .replace(/\\s+/g, " ").toLowerCase();
          const hit = items.find((el) => el.offsetParent !== null && !el.disabled &&
            text(el).includes(want));
          if (hit) hit.click();
          return !!hit;
        })()`);
        log.push(`click ${step.click}: ${found ? "ok" : "NOT FOUND"}`);
        await sleep(300);
      } else if ("type" in step) {
        const [selector, text] = step.type;
        const ok = await js(`(() => {
          const el = document.querySelector(${JSON.stringify(selector)});
          if (!el) return false;
          const setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el), "value").set;
          setter.call(el, ${JSON.stringify(text)});
          el.dispatchEvent(new Event("input", { bubbles: true }));
          return true;
        })()`);
        log.push(`type ${selector}: ${ok ? "ok" : "NOT FOUND"}`);
      } else if ("until" in step) {
        const limit = Date.now() + (step.timeout ?? 60_000);
        let seen = false;
        while (Date.now() < limit && !seen) {
          seen = await js(`document.body.innerText.includes(${JSON.stringify(step.until)})`);
          if (!seen) await sleep(250);
        }
        log.push(`until ${step.until}: ${seen ? "ok" : "TIMED OUT"}`);
      } else if ("shot" in step) {
        await sleep(200);
        const image = await window.webContents.capturePage();
        writeFileSync(join(folder, step.shot), image.toPNG());
        log.push(`shot ${step.shot}`);
      } else if ("run" in step) {
        log.push(`run: ${JSON.stringify(await js(step.run))}`);
      }
    } catch (error) {
      log.push(`error in ${JSON.stringify(step)}: ${String(error)}`);
    }
  }
  writeFileSync(join(folder, "capture.log"), log.join("\n") + "\n", "utf-8");
}
