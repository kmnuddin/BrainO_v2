import { resolve } from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "electron-vite";
import type { Plugin } from "vite";

/** The page may load only its own files: no remote scripts, styles, fonts or connections
 * (it talks to the engine through the main process). Added to the built page only, since the
 * development server's hot reload needs an inline script. */
const CSP =
  "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; " +
  "img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; worker-src 'self' blob:; " +
  "object-src 'none'; base-uri 'none'; form-action 'none'";

function contentSecurityPolicy(): Plugin {
  return {
    name: "braino-csp",
    apply: "build",
    transformIndexHtml: () => [
      {
        tag: "meta",
        attrs: { "http-equiv": "Content-Security-Policy", content: CSP },
        injectTo: "head-prepend",
      },
    ],
  };
}

export default defineConfig({
  main: {
    build: { outDir: "out/main" },
  },
  preload: {
    build: {
      outDir: "out/preload",
      // A sandboxed preload cannot load ES modules.
      rollupOptions: { output: { format: "cjs", entryFileNames: "[name].cjs" } },
    },
  },
  renderer: {
    root: resolve("src/renderer"),
    build: {
      outDir: resolve("out/renderer"),
      rollupOptions: { input: resolve("src/renderer/index.html") },
    },
    plugins: [react(), contentSecurityPolicy()],
  },
});
