import { $ } from "bun";

// Read from current environment which Bun automatically populates via .env/.env.local
const apiUrl = process.env.API_URL || "http://127.0.0.1:8000";

await Bun.build({
  entrypoints: [
    "./src/sidepanel.ts",
    "./src/background.ts",
    "./src/content.ts",
    "./src/ml-worker.ts"
  ],
  outdir: "./dist",
  define: {
    "process.env.API_URL": JSON.stringify(apiUrl),
  },
});

console.log("✅ Extension built successfully. API_URL injected ->", apiUrl);

// Specifically for Chrome Extensions using transformers.js, we MUST copy the ONNX runtime webassembly binaries
// into our dist folder, because Chrome Manifest V3 CSP strictly prohibits dynamically importing blobs or fetching WASM from external CDNs.

await $`cp node_modules/onnxruntime-web/dist/*.wasm ./dist/`;
await $`cp node_modules/onnxruntime-web/dist/*.mjs ./dist/`;
console.log("✅ ONNX WASM binaries seamlessly copied to ./dist/ for CSP compliance.");
