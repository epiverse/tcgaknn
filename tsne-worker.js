// tsne-worker.js
// Runs t-SNE inside a Web Worker to avoid blocking the main thread.

// Import the tsne library inside the worker scope. Use CDN UMD build.
try {
    importScripts('https://cdn.jsdelivr.net/npm/tsne@1.0.1/tsne.js');
} catch (e) {
    // importScripts may fail in some environments; we'll detect later
}

self.onmessage = function(e) {
    const { id, embeddings, components, iterations } = e.data;

    // Helper to send error
    function postError(err) {
        self.postMessage({ id, error: String(err) });
    }

    // Detect constructor
    let TSNEConstructor = null;
    try {
        if (self.tsnejs && typeof self.tsnejs.tSNE === 'function') TSNEConstructor = self.tsnejs.tSNE;
        else if (typeof self.tSNE === 'function') TSNEConstructor = self.tSNE;
        else if (typeof self.TSNE === 'function') TSNEConstructor = self.TSNE;
        else if (self.tsne && typeof self.tsne === 'function') TSNEConstructor = self.tsne;
    } catch (err) {
        TSNEConstructor = null;
    }

    if (!TSNEConstructor) {
        postError('t-SNE constructor not found in worker after importScripts');
        return;
    }

    try {
        const dims = Math.max(1, Number(components) || 2);
        const iters = Math.max(10, typeof iterations === 'number' ? iterations : 200);

        const tsne = new TSNEConstructor({ dim: dims, perplexity: Math.max(5, Math.min(40, Math.floor(embeddings.length / 5))), theta: 0.5, iterations: iters, eta: 200 });
        tsne.initDataRaw(embeddings);
        const progressInterval = Math.max(1, Math.floor(iters / 20));
        for (let k = 0; k < iters; k++) {
            tsne.step();
            if (k % progressInterval === 0) {
                const percent = Math.floor((k / iters) * 100);
                // send lightweight progress updates so main thread can reset timeouts
                try { self.postMessage({ id, progress: percent }); } catch (e) { /* noop */ }
            }
        }

        const solution = tsne.getSolution();
        self.postMessage({ id, solution });
    } catch (err) {
        postError(err);
    }
};
