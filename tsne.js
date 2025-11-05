// tsne.js - offload t-SNE computation to a Web Worker when available

async function applyTSNE(embeddings, components, iterations) {
    // Basic validation
    if (!Array.isArray(embeddings) || embeddings.length === 0) return embeddings.map(e => e.slice(0, Math.max(1, Number(components) || 2)));

    const dims = Math.max(1, Number(components) || 2);
    const iters = Math.max(10, typeof iterations === 'number' ? iterations : 200);
    // Apply a short disclaimer: we use a fast random projection for t-SNE to speed up runs.
    // This only affects t-SNE (not UMAP/PCA) and is an approximation to reduce runtime.
    let usedProjection = false;
    try {
        if (typeof window.setStatus === 'function') {
            window.setStatus('Applying fast projection for t-SNE (speed mode)...');
        }
    } catch (e) {}

    // Pre-project to lower dimensions for much faster t-SNE while preserving distances
    const PROJECT_DIMS = 40;
    let embeddingsForTSNE = embeddings;
    try {
        if (embeddings && embeddings[0] && embeddings[0].length > PROJECT_DIMS && typeof window.randomProject === 'function') {
            embeddingsForTSNE = window.randomProject(embeddings, PROJECT_DIMS);
            usedProjection = true;
            console.info('t-SNE: using random projection to', PROJECT_DIMS, 'dimensions for speed.');
            try {
                const note = document.getElementById('speed-mode-note');
                if (note) note.style.display = 'block';
            } catch (e) {}
        }
    } catch (e) {
        console.warn('Random projection failed, proceeding with original embeddings', e);
        embeddingsForTSNE = embeddings;
    }

    // Adapt iterations based on dataset size to reduce runtime for large n
    const nSamples = embeddingsForTSNE.length;
    let adjustedIters = iters;
    if (nSamples > 2000) adjustedIters = Math.min(adjustedIters, 300);
    if (nSamples > 5000) adjustedIters = Math.min(adjustedIters, 200);

    // If Workers are available, use tsne-worker.js to compute off-thread
    if (typeof Worker !== 'undefined') {
        return await new Promise((resolve) => {
            const worker = new Worker('tsne-worker.js');
            const id = Math.random().toString(36).slice(2);
            // Start a timeout which will be reset each time the worker sends progress
            let timeout = setTimeout(() => {
                worker.terminate();
                console.warn('t-SNE worker timed out; falling back to main-thread computation.');
                resolve(embeddings.map(e => e.slice(0, dims)));
            }, 180 * 1000); // 180s timeout

            worker.onmessage = function(ev) {
                const data = ev.data;
                if (data.progress !== undefined) {
                    // reset the timeout when progress is reported
                    clearTimeout(timeout);
                    timeout = setTimeout(() => {
                        worker.terminate();
                        console.warn('t-SNE worker timed out; falling back to main-thread computation.');
                        resolve(embeddings.map(e => e.slice(0, dims)));
                    }, 180 * 1000);
                    // update UI status if available
                    try { if (typeof window.setStatus === 'function') window.setStatus(`t-SNE: ${data.progress}%`); } catch (e) {}
                    return;
                }

                if (data.error) {
                    clearTimeout(timeout);
                    console.warn('t-SNE worker error:', data.error);
                    worker.terminate();
                    resolve(embeddings.map(e => e.slice(0, dims)));
                } else if (data.solution) {
                    clearTimeout(timeout);
                    worker.terminate();
                    // If we projected, emit a short note in status and hide the disclaimer after a moment
                    try { if (usedProjection && typeof window.setStatus === 'function') window.setStatus('t-SNE (speed mode) complete'); } catch (e) {}
                    try { const note = document.getElementById('speed-mode-note'); if (note) setTimeout(() => note.style.display = 'none', 3000); } catch (e) {}
                    resolve(data.solution);
                }
            };

            worker.postMessage({ id, embeddings: embeddingsForTSNE, components: dims, iterations: adjustedIters });
        });
    }

    // No Worker support: fall back to main-thread (rare). Try to detect existing tsne constructors.
    // We intentionally keep this minimal; heavy workloads should use Workers.
    try {
        // Attempt to find constructor on window
        let TSNEConstructor = null;
        if (window.tsnejs && typeof window.tsnejs.tSNE === 'function') TSNEConstructor = window.tsnejs.tSNE;
        else if (typeof window.tSNE === 'function') TSNEConstructor = window.tSNE;
        else if (typeof window.TSNE === 'function') TSNEConstructor = window.TSNE;
        else if (window.tsne && typeof window.tsne === 'function') TSNEConstructor = window.tsne;

        if (!TSNEConstructor) {
            console.error('t-SNE constructor not found on main thread. Returning sliced embeddings.');
            return embeddings.map(e => e.slice(0, dims));
        }

        const tsne = new TSNEConstructor({ dim: dims, perplexity: Math.max(5, Math.min(50, Math.floor(embeddings.length / 5))), theta: 0.5, iterations: iters, eta: 200 });
        tsne.initDataRaw(embeddings);
        for (let k = 0; k < iters; k++) tsne.step();
        return tsne.getSolution();
    } catch (err) {
        console.error('Error running t-SNE on main thread:', err);
        return embeddings.map(e => e.slice(0, dims));
    }
}

window.applyTSNE = applyTSNE;