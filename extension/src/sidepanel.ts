declare var process: any;

// Apply theme instantly on script parse
if (localStorage.getItem('theme') === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
}

document.addEventListener('DOMContentLoaded', () => {
    const textInput = document.getElementById('text-input') as HTMLTextAreaElement;
    const analyzeBtn = document.getElementById('analyze-btn') as HTMLButtonElement;
    
    // Engine State
    let currentEngine: 'edge' | 'cloud' = 'edge';
    const engineStatus = document.getElementById('engine-status') as HTMLSpanElement;
    const engineSwitchBtn = document.getElementById('engine-switch-btn') as HTMLAnchorElement;

    const resultContainer = document.getElementById('result-container') as HTMLDivElement;
    const toxicityLabel = document.getElementById('toxicity-label') as HTMLSpanElement;
    const confidenceBarFill = document.getElementById('confidence-bar-fill') as HTMLDivElement;
    const confidenceText = document.getElementById('confidence-text') as HTMLSpanElement;
    const errorMessage = document.getElementById('error-message') as HTMLDivElement;
    const loader = document.getElementById('loader') as HTMLDivElement;
    const btnWhy = document.getElementById('btn-why') as HTMLButtonElement;
    const explanationContainer = document.getElementById('explanation-container') as HTMLDivElement;

    // Theme logic
    const themeToggleBtn = document.getElementById('theme-toggle') as HTMLButtonElement;
    const themeIcon = document.getElementById('theme-icon') as unknown as SVGSVGElement;

    const moonIcon = `<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>`;
    const sunIcon = `<circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>`;

    function updateThemeIcon() {
        if (document.documentElement.getAttribute('data-theme') === 'light') {
            themeIcon.innerHTML = sunIcon;
        } else {
            themeIcon.innerHTML = moonIcon;
        }
    }

    updateThemeIcon();

    themeToggleBtn.addEventListener('click', () => {
        let currentTheme = document.documentElement.getAttribute('data-theme');
        let newTheme = currentTheme === 'light' ? 'dark' : 'light';
        
        if (newTheme === 'light') {
            document.documentElement.setAttribute('data-theme', 'light');
        } else {
            document.documentElement.removeAttribute('data-theme');
        }
        
        localStorage.setItem('theme', newTheme);
        updateThemeIcon();
    });

    // Bun automatically replaces process.env variables at build time if they exist in .env files
    const API_URL = process.env.API_URL || 'http://127.0.0.1:8000';

    // Handle Engine Switch Toggle
    engineSwitchBtn.addEventListener('click', (e) => {
        e.preventDefault();
        if (currentEngine === 'edge') {
            currentEngine = 'cloud';
            engineStatus.textContent = 'Engine: Remote (API)';
            engineSwitchBtn.textContent = 'Switch to Local Browser';
        } else {
            currentEngine = 'edge';
            engineStatus.textContent = 'Engine: Local (Browser)';
            engineSwitchBtn.textContent = 'Switch to Remote API';
        }
    });

    let currentText = '';

    function showError(msg: string) {
        errorMessage.textContent = msg;
        errorMessage.style.display = 'block';
        resultContainer.style.display = 'none';
        loader.style.display = 'none';
        analyzeBtn.disabled = false;
    }

    function hideError() {
        errorMessage.style.display = 'none';
    }

    let mlWorker: Worker | null = null;
    function getWorker() {
        if (!mlWorker) {
            mlWorker = new Worker('../dist/ml-worker.js', { type: 'module' });
            mlWorker.addEventListener('error', (err) => {
                console.error("Worker error:", err);
                showError("Worker initialization failed. Check console.");
            });
            mlWorker.addEventListener('message', (e) => {
                const data = e.data;
                console.log("Worker message:", data);
                if (data.status === 'init') {
                    loader.innerHTML = `Downloading model (${data.name})...<span class="dots"></span>`;
                } else if (data.status === 'progress') {
                    loader.innerHTML = `Downloading model (${Math.round(data.progress)}%)...<span class="dots"></span>`;
                } else if (data.status === 'ready' || data.status === 'done') {
                    loader.innerHTML = `Running Edge Inference...<span class="dots"></span>`;
                }
            });
        }
        return mlWorker;
    }

    function runEdgeInference(text: string): Promise<any> {
        return new Promise((resolve, reject) => {
            const worker = getWorker();
            
            const handleMessage = (e: MessageEvent) => {
                const data = e.data;
                if (data.status === 'complete') {
                    worker.removeEventListener('message', handleMessage);
                    resolve(data.result);
                } else if (data.status === 'error') {
                    worker.removeEventListener('message', handleMessage);
                    reject(new Error(data.error));
                }
            };
            
            worker.addEventListener('message', handleMessage);
            worker.postMessage({ type: 'CLASSIFY', text });
        });
    }

    async function analyzeText(text: string) {
        if (!text || text.trim() === "") {
            showError("Please enter some text.");
            return;
        }

        currentText = text;
        hideError();
        resultContainer.style.display = 'none';
        explanationContainer.style.display = 'none';
        explanationContainer.innerHTML = '';
        btnWhy.style.display = 'none';
        
        analyzeBtn.disabled = true;
        loader.innerHTML = 'Analyzing models<span class="dots"></span>';
        loader.style.display = 'block';

        try {
            const isEdge = currentEngine === 'edge';
            let isToxic = false;
            let conf = 0;

            if (isEdge) {
                const result = await runEdgeInference(text);
                isToxic = result.label === 'toxic';
                conf = Math.round(result.confidence * 100);
            } else {
                const response = await fetch(`${API_URL}/predict`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });

                if (!response.ok) throw new Error("Server error");
                const data = await response.json();
                
                isToxic = data.label.toLowerCase() === 'toxic';
                conf = Math.round(data.confidence * 100);
            }

            toxicityLabel.textContent = isToxic ? 'Toxic' : 'Safe';
            toxicityLabel.className = isToxic ? 'toxic' : 'safe';
            confidenceBarFill.style.backgroundColor = isToxic ? 'var(--danger)' : 'var(--success)';
            
            confidenceBarFill.style.width = '0%';
            
            loader.style.display = 'none';
            resultContainer.style.display = 'flex';
            btnWhy.style.display = 'block';
            btnWhy.textContent = "Why is this classified this way?";
            
            setTimeout(() => {
                confidenceBarFill.style.width = `${conf}%`;
                confidenceText.textContent = `${conf}%`;
            }, 50);

        } catch (err: any) {
            console.error(err);
            showError(`Failed to analyze: ${err.message}. Ensure backend is running.`);
        } finally {
            analyzeBtn.disabled = false;
        }
    }

    // Handle user manual click
    analyzeBtn.addEventListener('click', () => {
        analyzeText(textInput.value);
    });

    // Handle "Why?" button click
    btnWhy.addEventListener('click', async () => {
        btnWhy.disabled = true;
        btnWhy.textContent = "Querying Vector DB & LLM...";
        
        try {
            const response = await fetch(`${API_URL}/explain`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: currentText })
            });

            if (!response.ok) throw new Error("Server error");
            const data = await response.json();
            
            btnWhy.style.display = 'none';
            explanationContainer.innerHTML = `<strong>LLM Analysis:</strong><br/><br/>${data.explanation}`;
            explanationContainer.style.display = 'block';

        } catch (err: any) {
            btnWhy.textContent = "Error fetching explanation";
            btnWhy.disabled = false;
        }
    });

    // Initialize from contextMenu trigger (storage)
    chrome.storage.local.get(['queuedText'], (result) => {
        if (result.queuedText) {
            textInput.value = result.queuedText;
            analyzeText(result.queuedText);
            chrome.storage.local.remove('queuedText'); // Clear once parsed
        }
    });

    // Handle real-time messages from background/content script if sidepanel is already open
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
        if ((message.type === 'ANALYZE_TEXT' || message.type === 'AUTO_ANALYZE_TEXT') && message.text) {
            if (currentText !== message.text) {
                textInput.value = message.text;
                analyzeText(message.text);
            }
            sendResponse({ received: true });
        }
    });
});
