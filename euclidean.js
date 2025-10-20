/**
 * Calculates the Euclidean distance between two vectors.
 * @param {number[]} vecA
 * @param {number[]} vecB
 * @returns {number} The Euclidean distance.
 */
function euclideanDistance(vecA, vecB) {
    let sumSq = 0;
    for (let i = 0; i < vecA.length; i++) {
        sumSq += Math.pow(vecA[i] - vecB[i], 2);
    }
    return Math.sqrt(sumSq);
}