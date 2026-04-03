# ToxicShield

ToxicShield is a hybrid NLP application designed to detect toxicity in text highlights directly from your browser. It consists of a Chrome Extension frontend interface utilizing the new Chrome Side Panel API, and is capable of running a dual-inference strategy:

1. **Cloud Backend**: A fully-fledged FastAPI server powered by PyTorch transformer models.
2. **Local Edge AI**: A serverless, purely local inference mode running offline in your browser leveraging WebAssembly (`wasm`) and `transformers.js` via a dedicated offline Web Worker.

## Architecture & Diagrams

Detailed diagrams for the system architecture and sequence flows are organized in the [`docs/diagrams/`](./docs/diagrams/) folder:
- **[`architecture.wsd`](./docs/diagrams/architecture.wsd)**: General system architecture overview.
- **[`seq_diagram_cloud_ai.wsd`](./docs/diagrams/seq_diagram_cloud_ai.wsd)**: Sequence diagram for the remote FastAPI server application.
- **[`seq_diagram_edge_ai.wsd`](./docs/diagrams/seq_diagram_edge_ai.wsd)**: Sequence diagram reflecting the actual implementation of real-time browser Edge AI using ONNX webassembly fetching.

## Directory Structure

- **Backend (`/backend`)**: Python, FastAPI, Uvicorn, PyTorch, Hugging Face Transformers. Managed using `uv`.
- **Extension (`/extension`)**: Google Chrome Extension built with HTML, CSS, TypeScript. Managed using `bun`.

## Usage & Setup

### 1. Backend (Python/FastAPI)
The backend runs a local inference server using `uv`. Make sure you have `uv` installed.
```bash
cd backend
uv run uvicorn main:app --reload --port 8000
```
This will automatically download the PyTorch weights the first time you run it.

### 2. Chrome Extension (TypeScript/Bun)
Navigate to the `extension` folder to install dependencies and build the code using `bun`.

To build for local development (which uses `.env.local` to connect to your local backend):
```bash
cd extension
bun install
bun run build
```

To build for production (using `.env.production`):
```bash
cd extension
bun run build:prod
```

### 3. Loading the Extension into Chrome
1. Open Google Chrome and navigate to `chrome://extensions/`.
2. Enable **Developer mode** toggle in the top right corner.
3. Click **Load unpacked** and select the `/extension/dist` folder.
4. Pin the extension, click the extension icon to open the ToxicShield Side Panel, highlight some text on any web page, and start analyzing!
