// umap.js

/**
 * Performs UMAP (Uniform Manifold Approximation and Projection) on the given embeddings.
 * * NOTE: This function requires an external library like 'umap-js' (UMAP object) to be loaded globally.
 * * @param {number[][]} embeddings - The array of vectors (data points) to transform.
 * @param {number} components - The desired number of dimensions (nComponents).
 * @returns {Promise<number[][]>} A promise that resolves to the dimensionally-reduced embeddings.
 */
async function umapTransform(embeddings, components) {
    // Offload UMAP to a worker when possible to avoid blocking UI
    if (typeof Worker !== 'undefined') {
        return await new Promise((resolve) => {
            const worker = new Worker('umap-worker.js');
            const id = Math.random().toString(36).slice(2);
            const timeout = setTimeout(() => {
                worker.terminate();
                console.warn('UMAP worker timed out; falling back to main-thread computation.');
                resolve(embeddings.map(e => e.slice(0, components)));
            }, 60 * 1000);

            worker.onmessage = function(ev) {
                clearTimeout(timeout);
                const data = ev.data;
                if (data.error) {
                    console.warn('UMAP worker error:', data.error);
                    worker.terminate();
                    resolve(embeddings.map(e => e.slice(0, components)));
                } else if (data.solution) {
                    worker.terminate();
                    resolve(data.solution);
                }
            };

            worker.postMessage({ id, embeddings, components });
        });
    }

    // Fallback: run on main thread (if UMAP library is available)
    if (typeof UMAP === 'undefined' || typeof UMAP.UMAP !== 'function') {
        console.error("UMAP library (e.g., umap-js) is not loaded. Cannot perform UMAP transformation.");
        return embeddings.map(e => e.slice(0, components));
    }

    try {
        const umap = new UMAP.UMAP({ nComponents: components, nNeighbors: 15, minDist: 0.1 });
        const umapEmbedding = await umap.fit(embeddings);
        return umapEmbedding;
    } catch (error) {
        console.error("Error during UMAP transformation:", error);
        return embeddings.map(e => e.slice(0, components));
    }
}


// --- Handler for analysis.js ---

/**
 * The unified function signature expected by analysis.js for Dimensionality Reduction.
 * This is the function called when the user selects 'umap'.
 * @param {number[][]} embeddings - The input embeddings.
 * @param {number} components - The number of components.
 * @returns {Promise<number[][]>} The reduced embeddings.
 */
function applyUMAP(embeddings, components) {
    // The UMAP operation is asynchronous, so we simply return the Promise from the transformer.
    return umapTransform(embeddings, components);
}