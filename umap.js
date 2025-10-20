// umap.js

/**
 * Performs UMAP (Uniform Manifold Approximation and Projection) on the given embeddings.
 * * NOTE: This function requires an external library like 'umap-js' (UMAP object) to be loaded globally.
 * * @param {number[][]} embeddings - The array of vectors (data points) to transform.
 * @param {number} components - The desired number of dimensions (nComponents).
 * @returns {Promise<number[][]>} A promise that resolves to the dimensionally-reduced embeddings.
 */
async function umapTransform(embeddings, components) {
    if (typeof UMAP === 'undefined' || typeof UMAP.UMAP !== 'function') {
        console.error("UMAP library (e.g., umap-js) is not loaded. Cannot perform UMAP transformation.");
        // Fallback to slicing the original data, ensuring async contract is met
        return embeddings.map(e => e.slice(0, components));
    }

    try {
        // Initialize UMAP with user-defined components and reasonable defaults
        const umap = new UMAP.UMAP({
            nComponents: components,
            nNeighbors: 15,
            minDist: 0.1,
            // You may need to adjust the following for performance/quality:
            // repulsionStrength: 1.0,
            // setOperationStrength: 1.0,
            // negativeSampleRate: 5,
        });

        console.log(`Computing UMAP embeddings to ${components} components...`);
        // The fit method is asynchronous in umap-js
        const umapEmbedding = await umap.fit(embeddings);
        console.log("UMAP embedding computation completed.");

        return umapEmbedding;
    } catch (error) {
        console.error("Error during UMAP transformation:", error);
        // Fallback to slicing the original data
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