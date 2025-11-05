// umap-worker.js
// Runs UMAP inside a Web Worker to avoid blocking the main thread.

// Import umap-js inside the worker
try {
    importScripts('https://cdn.jsdelivr.net/npm/umap-js@1.4.0/lib/umap-js.min.js');
} catch (e) {
    // ignore; we'll detect below
}

self.onmessage = async function(e) {
    const { id, embeddings, components } = e.data;

    function postError(err) {
        self.postMessage({ id, error: String(err) });
    }

    if (typeof self.UMAP === 'undefined' || typeof self.UMAP.UMAP !== 'function') {
        postError('UMAP library not available inside worker');
        return;
    }

    try {
        const umap = new self.UMAP.UMAP({ nComponents: components, nNeighbors: 15, minDist: 0.1 });
        const res = await umap.fit(embeddings);
        self.postMessage({ id, solution: res });
    } catch (err) {
        postError(err);
    }
};
