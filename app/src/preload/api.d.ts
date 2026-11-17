import type { BrainoBridge } from "../shared/ipc";

declare global {
  interface Window {
    braino: BrainoBridge;
  }
}

export {};
