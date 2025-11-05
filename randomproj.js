// randomproj.js
// Fast Johnson-Lindenstrauss random projection for quick dimensionality reduction.
// This is a lightweight approximation used to speed up t-SNE preprocessing.

function gaussianRandom() {
    // Box-Muller transform
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/**
 * Projects `embeddings` (Array of Array<number>) to `targetDim` using a random Gaussian matrix.
 * Returns a new array with Float32Array rows.
 */
function randomProject(embeddings, targetDim = 40) {
    if (!Array.isArray(embeddings) || embeddings.length === 0) return embeddings;
    const n = embeddings.length;
    const d = embeddings[0].length;
    if (d <= targetDim) return embeddings.map(r => Float32Array.from(r));

    // Create random projection matrix R of shape d x targetDim
    const R = new Float32Array(d * targetDim);
    // Fill with N(0,1/targetDim) to preserve variance roughly
    const scale = 1.0 / Math.sqrt(targetDim);
    for (let i = 0; i < R.length; i++) R[i] = gaussianRandom() * scale;

    const out = new Array(n);
    for (let i = 0; i < n; i++) {
        const row = new Float32Array(targetDim);
        const emb = embeddings[i];
        for (let j = 0; j < d; j++) {
            const v = emb[j];
            const base = j * targetDim;
            for (let k = 0; k < targetDim; k++) {
                row[k] += v * R[base + k];
            }
        }
        out[i] = Array.from(row);
    }

    return out;
}

window.randomProject = randomProject;
