import { pipeline, env } from '@huggingface/transformers';

// Skip local checks, aggressively load from Hugging Face Hub
env.allowLocalModels = false;

if (env.backends?.onnx?.wasm) {
    env.backends.onnx.wasm.numThreads = 1;
    env.backends.onnx.wasm.wasmPaths = '/dist/';
}

class PipelineSingleton {
  static task: any = 'text-classification';
  static model = 'Xenova/toxic-bert';
  static instance: any = null;

  static async getInstance(progress_callback: any) {
    if (this.instance === null) {
      // Initialize the pipeline
      this.instance = await pipeline(this.task, this.model, { 
          progress_callback 
      });
    }
    return this.instance;
  }
}

// Listen for messages from the Side Panel
self.addEventListener('message', async (event: MessageEvent) => {
  if (event.data.type !== 'CLASSIFY') return;
  
  const text = event.data.text;
  
  try {
    // We pass a callback function to send progress updates to the UI
    const classifier = await PipelineSingleton.getInstance((x: any) => {
        // x contains { status: 'init' | 'progress' | 'done', name: string, progress: number }
        self.postMessage(x);
    });
    
    // Perform inference. Toxic-bert returns all labels by default if topk=null
    const output = await classifier(text, { topk: null });
    
    // Toxic-bert outputs an array of objects for multiple toxicity traits
    // Look for the 'toxic' trait score.
    const toxicLabel = output.find((o: any) => o.label === 'toxic');
    const toxicScore = toxicLabel ? toxicLabel.score : 0;
    
    // Format perfectly to match what our FastAPI backend returns!
    const isToxic = toxicScore > 0.5;
    
    self.postMessage({
      status: 'complete',
      result: {
          label: isToxic ? 'toxic' : 'safe',
          confidence: isToxic ? toxicScore : (1 - toxicScore) // If safe, confidence is inverted
      }
    });

  } catch (err: any) {
    console.error(err);
    self.postMessage({
      status: 'error',
      error: err.message
    });
  }
});
