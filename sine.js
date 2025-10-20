/**
 * Calculates the Sine distance, which is commonly interpreted as the
 * Angular Distance (1 - Cosine Similarity) in vector space analysis.
 * It simply calls the cosineDistance function defined in cosine.js.
 * @param {number[]} vecA
 * @param {number[]} vecB
 * @returns {number} The Sine/Angular distance.
 */
function sineDistance(vecA, vecB) {
    // Requires cosine.js to be loaded before this file
    return cosineDistance(vecA, vecB);
}