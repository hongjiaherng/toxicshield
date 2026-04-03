chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(console.error);

chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "analyze-toxicity",
        title: "Analyze Toxicity",
        contexts: ["selection"]
    });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
    if (info.menuItemId === "analyze-toxicity" && info.selectionText) {
        if (!tab || !tab.windowId) return;
        
        let selectedText = info.selectionText;

        // Open the side panel explicitly
        await chrome.sidePanel.open({ windowId: tab.windowId });
        
        // Save the queued text to storage so the sidepanel can pick it up immediately when it loads
        chrome.storage.local.set({ queuedText: selectedText }, () => {
            // Also attempt to send message in case it's already open
            chrome.runtime.sendMessage({ type: 'ANALYZE_TEXT', text: selectedText }).catch(err => {
                // Ignore error if side panel is starting up, it will gracefully read from storage.
            });
        });
    }
});
