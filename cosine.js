/**
 * Calculates the Cosine distance (1 - Cosine Similarity) between two vectors.
 * @param {number[]} vecA
 * @param {number[]} vecB
 * @returns {number} The Cosine distance.
 */
function cosineDistance(vecA, vecB) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    const magnitude = Math.sqrt(normA) * Math.sqrt(normB);

    // Avoid division by zero if magnitudes are zero
    if (magnitude === 0) return 1;

    const similarity = dotProduct / magnitude;
    return 1 - similarity; // Cosine Distance
}