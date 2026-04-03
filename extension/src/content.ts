document.addEventListener('mouseup', () => {
    // We only use mouseup so we don't spam the background page every time the selection shifts
    // while the user is actively dragging their mouse. 
    // Set a tiny timeout to ensure the selection has registered.
    setTimeout(() => {
        const selection = window.getSelection()?.toString().trim();
        if (selection) {
            // Attempt to fire to the side panel
            chrome.runtime.sendMessage({ type: 'AUTO_ANALYZE_TEXT', text: selection }).catch(() => {
                // Ignore errors. This simply means the Chrome SidePanel is not currently open.
            });
        }
    }, 100);
});
