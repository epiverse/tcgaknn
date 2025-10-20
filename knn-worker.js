// KNN Worker Implementation
class MaxHeap {
    constructor() {
        this.heap = [];
    }

    size() {
        return this.heap.length;
    }

    push(item) {
        this.heap.push(item);
        this._bubbleUp(this.heap.length - 1);
    }

    pop() {
        if (this.heap.length === 0) return null;
        const result = this.heap[0];
        const last = this.heap.pop();
        if (this.heap.length > 0) {
            this.heap[0] = last;
            this._bubbleDown(0);
        }
        return result;
    }

    peek() {
        return this.heap[0] || null;
    }

    _bubbleUp(index) {
        while (index > 0) {
            const parentIndex = Math.floor((index - 1) / 2);
            if (this.heap[index].dist > this.heap[parentIndex].dist) {
                [this.heap[index], this.heap[parentIndex]] = [this.heap[parentIndex], this.heap[index]];
                index = parentIndex;
            } else {
                break;
            }
        }
    }

    _bubbleDown(index) {
        while (true) {
            let largest = index;
            const leftChild = 2 * index + 1;
            const rightChild = 2 * index + 2;

            if (leftChild < this.heap.length && this.heap[leftChild].dist > this.heap[largest].dist) {
                largest = leftChild;
            }
            if (rightChild < this.heap.length && this.heap[rightChild].dist > this.heap[largest].dist) {
                largest = rightChild;
            }

            if (largest === index) break;

            [this.heap[index], this.heap[largest]] = [this.heap[largest], this.heap[index]];
            index = largest;
        }
    }
}

// Distance metric implementations
function euclideanDistance(a, b, aNorm, bNorm) {
    if (aNorm !== undefined && bNorm !== undefined) {
        let dotProduct = 0;
        for (let i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
        }
        return Math.sqrt(Math.max(0, aNorm + bNorm - 2 * dotProduct));
    }

    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        const diff = a[i] - b[i];
        sum += diff * diff;
    }
    return Math.sqrt(sum);
}

function cosineDistance(a, b) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    const similarity = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    return 1 - similarity;
}

function sineDistance(a, b) {
    return 1 - Math.abs(cosineDistance(a, b) - 1);
}

function getDistanceFunction(metric) {
    if (metric === 'euclidean') return euclideanDistance;
    if (metric === 'cosine') return cosineDistance;
    if (metric === 'sine') return sineDistance;
    throw new Error(`Unknown metric: ${metric}`);
}

// Process a chunk of queries
function processChunk(data) {
    const {
        startIdx,
        endIdx,
        embeddings,
        ids,
        K,
        metric,
        squaredNorms
    } = data;

    const results = {};
    const distanceFn = getDistanceFunction(metric);

    for (let i = startIdx; i < endIdx; i++) {
        const queryId = ids[i];
        const queryEmb = embeddings[i];
        const queryNorm = squaredNorms ? squaredNorms[i] : undefined;

        const heap = new MaxHeap();

        for (let j = 0; j < embeddings.length; j++) {
            if (i === j) continue;

            const dist = metric === 'euclidean' && squaredNorms
                ? distanceFn(queryEmb, embeddings[j], queryNorm, squaredNorms[j])
                : distanceFn(queryEmb, embeddings[j]);

            if (heap.size() < K) {
                heap.push({ id: ids[j], dist });
            } else if (dist < heap.peek().dist) {
                heap.pop();
                heap.push({ id: ids[j], dist });
            }
        }

        const neighbors = [];
        while (heap.size() > 0) {
            neighbors.unshift(heap.pop().id);
        }
        results[queryId] = neighbors;

        // Report progress for this item
        const progress = Math.round(((i - startIdx + 1) / (endIdx - startIdx)) * 100);
        self.postMessage({
            type: 'progress',
            progress,
            workerId: data.workerId,
            total: endIdx - startIdx
        });
    }

    // Send back results
    self.postMessage({
        type: 'complete',
        results,
        workerId: data.workerId
    });
}

// Handle messages from the main thread
self.onmessage = function(e) {
    if (e.data.type === 'process') {
        processChunk(e.data);
    }
};