// tsne.js

/**
 * Performs t-SNE (t-distributed Stochastic Neighbor Embedding) on the given embeddings.
 * This function is an ASYNCHRONOUS wrapper for the iterative t-SNE process.
 * NOTE: This requires the tsnejs library to be loaded globally (window.tsnejs).
 * * @param {number[][]} embeddings - The array of vectors (data points) to transform.
 * @param {number} components - The desired number of dimensions (dim). This is typically 2 or 3.
 * @param {number} iterations - The total number of steps to run t-SNE. Taken from the 'Components/Iterations' input.
 * @returns {Promise<number[][]>} A promise that resolves to the dimensionally-reduced embeddings.
 */
async function tsneTransform(embeddings, components, iterations) {
    if (typeof window.tsnejs === 'undefined' || typeof window.tsnejs.tSNE !== 'function') {
        console.error("tsnejs library is not loaded. Cannot perform t-SNE transformation.");
        // Fallback to slicing the original data
        return embeddings.map(e => e.slice(0, components));
    }

    // t-SNE has a minimum perplexity requirement based on the data size
    const perplexity = Math.max(1, Math.min(30, Math.floor(embeddings.length / 5)));

    // Ensure iterations are sensible
    const actualIterations = Math.max(10, iterations);

    try {
        const tsne = new window.tsnejs.tSNE({
            dim: components, // The dimension to reduce to
            perplexity: perplexity,
            theta: 0.5,
            iterations: actualIterations,
            eta: 200
        });

        console.log(`Computing t-SNE embeddings to ${components} components over ${actualIterations} iterations...`);

        // Initialize tSNE with the input vectors
        tsne.initDataRaw(embeddings);

        // Perform t-SNE steps iteratively. We use a Promise to make this synchronous-looking
        // while allowing the browser to remain responsive (though tsnejs is compute-heavy).
        // Since tsnejs doesn't natively support async/await, we run it in a single loop,
        // which will block the main thread for the duration of the calculation.
        // A robust solution would use a Web Worker, but for this structure, a simple loop suffices.
        for (let k = 0; k < actualIterations; k++) {
            tsne.step();

            // Update progress every 5% or every 10 iterations (whichever is larger)
            if (k % Math.max(10, Math.floor(actualIterations / 20)) === 0) {
                const percent = Math.floor((k / actualIterations) * 100);
                if (typeof window.setStatus === 'function') {
                    try { window.setStatus(`t-SNE: ${percent}%`); } catch (e) { /* noop */ }
                }
                // Yield to allow UI updates
                await new Promise(resolve => requestAnimationFrame(resolve));
            }
        }

        // Ensure 100% status
        if (typeof window.setStatus === 'function') {
            try { window.setStatus('t-SNE: 100%'); } catch (e) { /* noop */ }
        }

        console.log("t-SNE computation completed.");

        // Return the final result
        return tsne.getSolution();

    } catch (error) {
        console.error("Error during t-SNE transformation:", error);
        // Fallback to slicing the original data
        return embeddings.map(e => e.slice(0, components));
    }
}


// --- Handler for analysis.js ---

/**
 * The unified function signature expected by analysis.js for Dimensionality Reduction.
 * This is the function called when the user selects 'tsne'.
 * * NOTE: The 'components' input in the HTML is used for BOTH the target dimension
 * (dim) and the number of iterative steps (iterations) in this implementation,
 * as the 'Components/Iterations' label suggests it serves a dual purpose for t-SNE.
 * * @param {number[][]} embeddings - The input embeddings.
 * @param {number} components - The value from the 'Components/Iterations' input.
 * @returns {Promise<number[][]>} The reduced embeddings.
 */
function applyTSNE(embeddings, components) {
    // We arbitrarily set the target output dimension to 2 for visualization
    // unless the user specifies a value of 3 or more in the components box,
    // but the iteration count is set by the input value.
    // Given the UI shows '50', let's use a standard 2D output (common for t-SNE)
    // but use the input for iterations.

    // Safely assume the target dimension (dim) is 2 or 3 based on standard practice,
    // or you can explicitly use the input value:
    const outputDim = 2; // Typically 2D for t-SNE output

    // The 'components' input value (e.g., 50) is used as the number of iterations.
    const numIterations = components;

    // The t-SNE operation is asynchronous, so we return the Promise.
    return tsneTransform(embeddings, outputDim, numIterations);
}