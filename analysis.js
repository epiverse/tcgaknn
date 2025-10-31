// analysis.js

// 1. DATA LOADING AND SETUP
// Use the raw.githubusercontent.com URL to fetch the ZIP directly (avoids HTML redirects)
const EMBEDDINGS_ZIP_URL = "https://raw.githubusercontent.com/epiverse/tcgadata/main/TITAN/text_image_embeddings.tsv.zip";

// Helper function to convert the string representation of an embedding into a float array
function parseEmbedding(embStr) {
    // Handles array strings like "[0.1, 0.2, ...]"
    return embStr
        .replace(/[\[\]"]/g, '')
        .split(',')
        .map(s => parseFloat(s.trim()))
        .filter(n => !isNaN(n));
}

/**
 * Fetches, unzips, and parses the TSV data from the GitHub URL.
 * @returns {Promise<Array<Object>>} A promise that resolves to the parsed array of data objects.
 */
async function loadData() {
    console.log("Fetching embeddings ZIP file...");
    setStatus('Phase 1/5 (0%): Downloading data...');
    try {
        // 1. Fetch the ZIP file
        const response = await fetch(EMBEDDINGS_ZIP_URL);
        if (!response.ok) {
            throw new Error(`Failed to fetch embeddings file. HTTP Status: ${response.status}`);
        }

        const dataBuffer = await response.arrayBuffer();
    console.log("Successfully fetched ZIP file. Unzipping...");
    setStatus('Unzipping and parsing data...');

        // 2. Unzip the file
        // If JSZip isn't loaded yet, try to dynamically load a CDN copy before failing.
        if (typeof JSZip === 'undefined') {
            console.warn('JSZip not present. Attempting to load JSZip dynamically from CDN...');
            // Try jsDelivr CDN first
            await new Promise((resolve, reject) => {
                const script = document.createElement('script');
                script.src = 'https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js';
                script.onload = () => resolve();
                script.onerror = () => reject(new Error('Dynamic JSZip load failed'));
                document.head.appendChild(script);
                // Timeout after 8s
                setTimeout(() => reject(new Error('Timed out loading JSZip')), 8000);
            }).catch(err => {
                console.warn('Dynamic JSZip load failed:', err);
            });
        }

        if (typeof JSZip === 'undefined') {
             throw new Error("JSZip library is required but not loaded in HTML.");
        }

        const zip = await JSZip.loadAsync(dataBuffer);

        const fileName = 'text_image_embeddings.tsv';
        const file = zip.file(fileName);
        if (!file) {
            throw new Error(`File '${fileName}' not found in the ZIP archive.`);
        }

        // Get the file as a Uint8Array and stream-parse it in chunks to avoid
        // allocating a giant string or a large array of lines at once.
        console.log("Successfully fetched ZIP file. Parsing TSV entry as stream...");
        setStatus('Phase 2/5 (20%): Extracting ZIP...');
        const uint8array = await file.async('uint8array');

        // Stream parser that decodes chunks and processes completed lines.
        const decoder = new TextDecoder('utf-8');
        const CHUNK_SIZE = 64 * 1024; // 64KB
        let pos = 0;
        let carry = '';
        let headers = null;
        const data = [];
        let lastPercent = -1;

        setStatus('Phase 3/5 (40%): Parsing data - 0%');
        while (pos < uint8array.length) {
            const end = Math.min(pos + CHUNK_SIZE, uint8array.length);
            const chunk = uint8array.subarray(pos, end);
            pos = end;

            // Decode the current chunk and prepend any carry-over from the previous chunk
            let text = decoder.decode(chunk, { stream: true });
            text = carry + text;

            // Split into lines, keep the last partial line as carry
            const lines = text.split('\n');
            carry = lines.pop();

            for (let li = 0; li < lines.length; li++) {
                const rawLine = lines[li].replace(/\r$/, '');
                if (!rawLine) continue;

                if (!headers) {
                    headers = rawLine.split('\t');
                    continue;
                }

                const values = rawLine.split('\t');
                const row = {};
                headers.forEach((header, index) => {
                    row[header.trim()] = values[index] ? values[index].trim() : '';
                });

                data.push({
                    i: parseInt(row.i),
                    id: row.id,
                    patient_id: row.patient_id,
                    cancer_type: row.cancer_type,
                    text_embedding: parseEmbedding(row.embeddings),
                    image_embedding: parseEmbedding(row.image_embedding)
                });
            }

            // Update percent progress, but only when it changes to avoid UI thrash
            const percent = Math.floor((pos / uint8array.length) * 100);
            if (percent !== lastPercent) {
                lastPercent = percent;
                setStatus(`Phase 3/5 (40%): Parsing data - ${percent}%`);
                // Yield to the browser so it can repaint the UI and show the updated percent.
                // Using requestAnimationFrame is lighter than setTimeout(0) and schedules
                // the continuation after the next paint.
                await new Promise(resolve => requestAnimationFrame(resolve));
            }
        }

        // Process any remaining carry as the last line
        if (carry) {
            const lastLine = carry.replace(/\r$/, '');
            if (lastLine) {
                if (!headers) {
                    headers = lastLine.split('\t');
                } else {
                    const values = lastLine.split('\t');
                    const row = {};
                    headers.forEach((header, index) => {
                        row[header.trim()] = values[index] ? values[index].trim() : '';
                    });
                    data.push({
                        i: parseInt(row.i),
                        id: row.id,
                        patient_id: row.patient_id,
                        cancer_type: row.cancer_type,
                        text_embedding: parseEmbedding(row.embeddings),
                        image_embedding: parseEmbedding(row.image_embedding)
                    });
                }
            }
        }

        console.log(`Data parsing complete. Loaded ${data.length} samples.`);

        // Sanity-check: filter valid samples (both text and image embeddings present)
        const validData = data.filter(d => d.text_embedding.length > 0 && d.image_embedding.length > 0);
        const validCount = validData.length;
        // Print a short summary and the first valid sample for runtime inspection
        console.log('Sanity check after loadData():', {
            totalRows: data.length,
            validSamples: validCount,
            firstValidSample: validData.length ? validData[0] : null
        });

        setStatus('Phase 3/5 (60%): Data loaded');
        return validData;

    } catch (error) {
        console.error("Error during data loading:", error);
        setStatus(`Failed to load data: ${error.message}`);
        alert(`Failed to load data: ${error.message}`);
        return []; // Return an empty array on failure
    }
}

// Global variable to hold the loaded data (will be populated async)
let data = [];

// --- 2. DISTANCE METRIC HANDLER ---

// This function now relies on the globally available distance functions
function getDistanceFunction(metric) {
    if (metric === 'euclidean') return euclideanDistance;
    if (metric === 'cosine') return cosineDistance;
    if (metric === 'sine') return sineDistance;
    throw new Error(`Unknown metric: ${metric}`);
}

// --- 3. DIMENSIONALITY REDUCTION (DR) HANDLER ---

// This function is ASYNCHRONOUS to handle external library calls
async function applyDR(embeddings, model, components, iterations) {
    if (model === 'no_model') {
        return embeddings; // Return the original embeddings
    }
    if (model === 'pca') {
        return await applyPCA(embeddings, components);
    }
    if (model === 'umap') {
        return await applyUMAP(embeddings, components);
    }
    if (model === 'tsne') {
        // Pass components as output dimensionality and iterations as the t-SNE budget
        return await applyTSNE(embeddings, components, iterations);
    }

    return embeddings;
}

// --- 4. K-NEAREST NEIGHBORS (KNN) ---

// Helper function to pre-compute squared norms for Euclidean distance
function precomputeSquaredNorms(embeddings) {
    return embeddings.map(emb => emb.reduce((sum, val) => sum + val * val, 0));
}

// Worker pool management
class WorkerPool {
    constructor(numWorkers) {
        this.workers = [];
        this.workerStatus = new Map(); // Track worker status and progress
        this.results = new Map();      // Store results from workers
        this.progressCallback = null;   // Callback for progress updates

        // Create workers
        for (let i = 0; i < numWorkers; i++) {
            const worker = new Worker('knn-worker.js');
            worker.onmessage = (e) => this.handleWorkerMessage(e.data, i);
            this.workers.push(worker);
            this.workerStatus.set(i, { busy: false, progress: 0 });
        }
    }

    handleWorkerMessage(message, workerId) {
        if (message.type === 'progress') {
            this.workerStatus.get(workerId).progress = message.progress;
            if (this.progressCallback) {
                // Calculate overall progress across all workers
                const totalProgress = Array.from(this.workerStatus.values())
                    .reduce((sum, status) => sum + status.progress, 0);
                const overallProgress = Math.floor(totalProgress / this.workers.length);
                this.progressCallback(overallProgress);
            }
        } else if (message.type === 'complete') {
            this.results.set(workerId, message.results);
            this.workerStatus.get(workerId).busy = false;
            this.workerStatus.get(workerId).progress = 100;
        }
    }

    async processData(chunks) {
        this.results.clear();
        const promises = [];

        for (let i = 0; i < chunks.length; i++) {
            const workerId = i % this.workers.length;
            const worker = this.workers[workerId];

            // Add workerId to chunk data
            chunks[i].workerId = workerId;

            // Create promise for this chunk
            const promise = new Promise((resolve) => {
                const checkComplete = setInterval(() => {
                    if (!this.workerStatus.get(workerId).busy) {
                        clearInterval(checkComplete);
                        resolve(this.results.get(workerId));
                    }
                }, 100);
            });

            promises.push(promise);

            // Mark worker as busy and send data
            this.workerStatus.get(workerId).busy = true;
            this.workerStatus.get(workerId).progress = 0;
            worker.postMessage({ type: 'process', ...chunks[i] });
        }

        // Wait for all chunks to complete
        const results = await Promise.all(promises);

        // Combine results from all workers
        return results.reduce((acc, curr) => ({ ...acc, ...curr }), {});
    }

    setProgressCallback(callback) {
        this.progressCallback = callback;
    }

    terminate() {
        this.workers.forEach(worker => worker.terminate());
    }
}

// Main KNN function
async function getKNN(data, K, embeddingKey, drModel, components, iterations, metric) {
    const rawEmbeddings = data.map(d => d[embeddingKey]);

    if (embeddingKey === 'text_embedding') setStatus('Phase 4/5 (70%): Calculating text KNN...');
    if (embeddingKey === 'image_embedding') setStatus('Phase 4/5 (80%): Calculating image KNN...');

    await new Promise(resolve => requestAnimationFrame(resolve));

    const drPhase = embeddingKey === 'text_embedding' ? '70' : '80';
    let processedEmbeddings;

    if (drModel === 'no_model') {
        setStatus(`Phase 4/5 (${drPhase}%): Computing KNN on raw embeddings...`);
        processedEmbeddings = rawEmbeddings;
    } else {
        setStatus(`Phase 4/5 (${drPhase}%): Applying ${drModel.toUpperCase()}...`);
    await new Promise(resolve => requestAnimationFrame(resolve));
    processedEmbeddings = await applyDR(rawEmbeddings, drModel, components, iterations);
    }

    // Pre-compute squared norms for Euclidean distance
    const squaredNorms = metric === 'euclidean' ? precomputeSquaredNorms(processedEmbeddings) : null;

    setStatus(`Phase 4/5 (${drPhase}%): Computing nearest neighbors...`);
    await new Promise(resolve => requestAnimationFrame(resolve));

    // Initialize worker pool with number of workers based on CPU cores
    // navigator.hardwareConcurrency returns the number of logical cores
    const numWorkers = Math.max(2, Math.min(4, navigator.hardwareConcurrency || 4));
    const workerPool = new WorkerPool(numWorkers);

    // Prepare data chunks for workers
    const chunkSize = Math.ceil(processedEmbeddings.length / (numWorkers * 2));
    const chunks = [];

    for (let start = 0; start < processedEmbeddings.length; start += chunkSize) {
        const end = Math.min(start + chunkSize, processedEmbeddings.length);
        chunks.push({
            startIdx: start,
            endIdx: end,
            embeddings: processedEmbeddings,
            ids: data.map(d => d.id),
            K,
            metric,
            squaredNorms
        });
    }

    // Set up progress callback
    workerPool.setProgressCallback(progress => {
        setStatus(`Phase 4/5 (${drPhase}%): Computing nearest neighbors - ${progress}%`);
    });

    // Process data using worker pool
    const knnResults = await workerPool.processData(chunks);

    // Clean up workers
    workerPool.terminate();

    return knnResults;
}// --- 5. MAIN ANALYSIS FUNCTION ---

// This function is ASYNCHRONOUS because it calls loadData and getKNN
async function runAnalysis() {
    try {
        // Ensure data is loaded
        if (data.length === 0) {
            document.getElementById('knn-results').innerHTML = `<p>Loading data...</p>`;
            data = await loadData();
            if (data.length === 0) {
                document.getElementById('knn-results').innerHTML = `<p class="error">Data load failed or file was empty.</p>`;
                return;
            }
        }

        // Get user inputs
        const K = parseInt(document.getElementById('k-neighbors').value);

        const textDrModel = document.getElementById('text-dr-model').value;
        const textMetric = document.getElementById('text-metric').value;
        let textComponents = parseInt(document.getElementById('text-components').value);
        let textIterations = parseInt(document.getElementById('text-iterations').value);

        const imageDrModel = document.getElementById('image-dr-model').value;
        const imageMetric = document.getElementById('image-metric').value;
        let imageComponents = parseInt(document.getElementById('image-components').value);
        let imageIterations = parseInt(document.getElementById('image-iterations').value);

        // Ensure component counts default to 3 when a model is selected but the input is empty/invalid
        if (textDrModel !== 'no_model' && (isNaN(textComponents) || textComponents < 1)) textComponents = 3;
        if (imageDrModel !== 'no_model' && (isNaN(imageComponents) || imageComponents < 1)) imageComponents = 3;

        // Ensure iterations are sensible for t-SNE only
        if (textDrModel === 'tsne' && (isNaN(textIterations) || textIterations < 10)) textIterations = 200;
        if (imageDrModel === 'tsne' && (isNaN(imageIterations) || imageIterations < 10)) imageIterations = 200;

        if (isNaN(K) || K <= 0) {
            alert("K (Nearest Neighbors) must be a positive number.");
            return;
        }

    document.getElementById('knn-results').innerHTML = `<p>Calculating KNN and Overlap...</p>`;
    setStatus('Running models and calculating KNN...');


        // --- A. Calculate KNN for Text ---
        // Changed getKNN signature to pass explicit components value
    const textKNN = await getKNN(data, K, 'text_embedding', textDrModel, textComponents, textIterations, textMetric);

        // --- B. Calculate KNN for Image ---
        // Changed getKNN signature to pass explicit components value
    const imageKNN = await getKNN(data, K, 'image_embedding', imageDrModel, imageComponents, imageIterations, imageMetric);

    // --- C. Calculate Overlap per Sample ---
    setStatus('Phase 5/5 (90%): Calculating overlap between KNNs...');
        const overlapResults = data.map(sample => {
            const textNeighbors = new Set(textKNN[sample.id]);
            const imageNeighbors = new Set(imageKNN[sample.id]);

            let overlapCount = 0;
            for (const neighborId of textNeighbors) {
                if (imageNeighbors.has(neighborId)) {
                    overlapCount++;
                }
            }

            return {
                ...sample,
                text_neighbors: Array.from(textNeighbors),
                image_neighbors: Array.from(imageNeighbors),
                overlap_count: overlapCount
            };
        });

        // --- D. Calculate Average Overlap per Patient ---
        const patientData = {};
        overlapResults.forEach(res => {
            if (!patientData[res.patient_id]) {
                patientData[res.patient_id] = [];
            }
            patientData[res.patient_id].push(res.overlap_count);
        });

        let totalOverlapSum = 0;
        let totalPatientSamples = 0;
        let htmlOutput = '';

        for (const patient_id in patientData) {
            const counts = patientData[patient_id];
            const avgOverlap = counts.reduce((a, b) => a + b, 0) / counts.length;

            totalOverlapSum += counts.reduce((a, b) => a + b, 0);
            totalPatientSamples += counts.length;

            htmlOutput += `
                <div class="patient-card">
                    <strong>Patient ID: ${patient_id}</strong><br>
                    Avg Overlap (out of ${K}): <strong>${avgOverlap.toFixed(3)}</strong><br>

                </div>
            `;
        }

        const overallAverageOverlap = totalOverlapSum / totalPatientSamples;

    // --- E. Display Results ---
    setStatus('Phase 5/5 (100%): Analysis complete');
        document.getElementById('knn-results').innerHTML = htmlOutput;
        document.getElementById('average-overlap').innerHTML = `
            Overall Average Overlap of Nearest Neighbors (K=${K}): ${overallAverageOverlap.toFixed(4)}
        `;

        // Maintain a history of runs (persist in the window so it survives multiple calls)
        try {
            if (!window.runsHistory) window.runsHistory = [];

            const textLabel = `${textDrModel.toUpperCase()} / ${textMetric.toUpperCase()}`;
            const imageLabel = `${imageDrModel.toUpperCase()} / ${imageMetric.toUpperCase()}`;

            // Store this run's KNN maps along with labels and parameters
            const runEntry = {
                id: window.runsHistory.length + 1,
                K,
                textModel: textDrModel,
                textMetric: textMetric,
                textComponents: textComponents,
                imageModel: imageDrModel,
                imageMetric: imageMetric,
                imageComponents: imageComponents,
                // keep KNN maps for potential future use
                textKNN,
                imageKNN,
                // overall overlap value for this run
                overallOverlap: overallAverageOverlap
            };

            window.runsHistory.push(runEntry);

            // Helper: compute average overlap between two knn maps
            function computeAverageOverlap(knnA, knnB) {
                let sum = 0;
                let count = 0;
                for (const sample of data) {
                    const id = sample.id;
                    const listA = knnA[id] || [];
                    const listB = knnB[id] || [];
                    if (!listA.length && !listB.length) continue;
                    const setB = new Set(listB);
                    let inter = 0;
                    for (const x of listA) if (setB.has(x)) inter++;
                    sum += inter;
                    count++;
                }
                return count === 0 ? 0 : sum / count;
            }

            // Build tables grouped by K. For each K, create a matrix with rows=models*metrics and cols=models*metrics
            const runs = window.runsHistory;
            // Define ordering
            const models = ['no_model', 'pca', 'umap', 'tsne'];
            const metrics = ['euclidean', 'cosine', 'sine'];

            // Helper to render nice model names
            function displayModel(m) {
                if (m === 'no_model') return 'No Model';
                if (m === 'pca') return 'PCA';
                if (m === 'umap') return 'UMAP';
                if (m === 'tsne') return 't-SNE';
                return String(m).toUpperCase();
            }

            // Group runs by K only. For each K we render a full matrix with
            // rows = (No Model, PCA, UMAP, t-SNE) x (Euclidean, Cosine, Sine)
            // and columns the same. If multiple historical runs match the same
            // text/image model+metric combo we show the most recent run's value.
            const runsByK = {};
            for (const r of runs) {
                if (!runsByK[r.K]) runsByK[r.K] = [];
                runsByK[r.K].push(r);
            }

            const modelsFixed = ['no_model', 'pca', 'umap', 'tsne'];
            const metricsFixed = ['euclidean', 'cosine', 'sine'];

            let multiTableHtml = '';
            for (const kKey of Object.keys(runsByK).sort((a, b) => a - b)) {
                const group = runsByK[kKey];

                // Helper: find the most recent run matching the combo
                function findMostRecentRun(textModel, textMetric, imageModel, imageMetric) {
                    for (let i = group.length - 1; i >= 0; i--) {
                        const rt = group[i];
                        if (rt.textModel === textModel && rt.textMetric === textMetric && rt.imageModel === imageModel && rt.imageMetric === imageMetric) {
                            return rt;
                        }
                    }
                    return null;
                }

                let table = '<div class="crosstab-box" style="background:#fff; padding:12px; border-radius:6px; border:1px solid #e6e6e6; margin-bottom:12px;">';
                table += `<strong>Cross-tabulation for K=${kKey}</strong>`;
                table += '<table style="width:100%; border-collapse:collapse; margin-top:8px;">';

                // Header row
                table += '<thead><tr>';
                table += '<th style="border-bottom:1px solid #ddd; text-align:left; padding:6px;">Text \\ Image</th>';
                for (const cm of modelsFixed) {
                    for (const metric of metricsFixed) {
                        let compLabel = '';
                        if (cm === 'no_model') compLabel = ' (dim=768)';
                        table += `<th style="border-bottom:1px solid #ddd; text-align:center; padding:6px;">${displayModel(cm)}${compLabel}<br>${metric.toUpperCase()}</th>`;
                    }
                }
                table += '</tr></thead>';

                // Body rows
                table += '<tbody>';
                for (const rm of modelsFixed) {
                    for (const rmetric of metricsFixed) {
                        let rowLabel = displayModel(rm);
                        if (rm === 'no_model') rowLabel += ' (dim=768)';
                        table += `<tr><td style="padding:6px; border-bottom:1px solid #f0f0f0;">${rowLabel}<br>${rmetric.toUpperCase()}</td>`;

                        for (const cm of modelsFixed) {
                            for (const cmetric of metricsFixed) {
                                const runMatch = findMostRecentRun(rm, rmetric, cm, cmetric);
                                if (runMatch) {
                                    table += `<td style="padding:6px; text-align:center;">${Number(runMatch.overallOverlap).toFixed(6)}</td>`;
                                } else {
                                    table += `<td style="padding:6px; text-align:center; color:#999;">-</td>`;
                                }
                            }
                        }

                        table += '</tr>';
                    }
                }
                table += '</tbody></table>';
                table += '<div style="font-size:0.9em; color:#666; margin-top:8px;">Each table shows the most recent overall-overlap value for the given Text (rows) vs Image (cols) model+metric combos for this K. Missing runs are shown as &quot;-&quot;.</div>';
                table += '</div>';

                multiTableHtml += table;
            }

            // Debugging aids: print runs and generated HTML so we can see why
            // a table might not appear in the page (empty runs, errors, etc.).
            console.debug('crosstab: runs for K groups:', Object.keys(runsByK).length, 'groups');
            console.debug('crosstab: generated HTML length:', multiTableHtml ? multiTableHtml.length : 0);
            const ct = document.getElementById('crosstab-container');
            if (ct) {
                ct.innerHTML = multiTableHtml || '<div style="color:#666;">No cross-tab data yet. Run an analysis to populate tables for each K.</div>';
            }
        } catch (e) {
            console.warn('Failed to build crosstab history table:', e);
        }

    } catch (error) {
        document.getElementById('knn-results').innerHTML = `<p class="error">An error occurred during analysis: ${error.message}</p>`;
        console.error("Analysis Error:", error);
        setStatus(`Error: ${error.message}`);
    }
}

// Helper to update the status box
function setStatus(message) {
    const box = document.getElementById('status-box');
    const msg = document.getElementById('status-message');
    if (!box || !msg) return;
    msg.textContent = message;
    box.style.display = 'block';
}

// Add event listeners for component inputs to handle the t-SNE-specific logic
document.getElementById('text-dr-model').addEventListener('change', function() {
    const componentsInput = document.getElementById('text-components');
    const iterationsInput = document.getElementById('text-iterations');
    const model = this.value;

    if (model === 'no_model') {
        componentsInput.disabled = true;
        componentsInput.value = '';
        iterationsInput.disabled = true;
        iterationsInput.value = '';
    } else if (model === 'tsne') {
        // t-SNE: allow setting both output dims and iterations
        componentsInput.disabled = false;
        if (!componentsInput.value) componentsInput.value = '3';
        iterationsInput.disabled = false;
        if (!iterationsInput.value) iterationsInput.value = '200';
    } else {
        // PCA/UMAP: allow setting output dims only
        componentsInput.disabled = false;
        if (!componentsInput.value) componentsInput.value = '3';
        iterationsInput.disabled = true;
        iterationsInput.value = '';
    }
});

document.getElementById('image-dr-model').addEventListener('change', function() {
    const componentsInput = document.getElementById('image-components');
    const iterationsInput = document.getElementById('image-iterations');
    const model = this.value;

    if (model === 'no_model') {
        componentsInput.disabled = true;
        componentsInput.value = '';
        iterationsInput.disabled = true;
        iterationsInput.value = '';
    } else if (model === 'tsne') {
        // t-SNE: allow setting both output dims and iterations
        componentsInput.disabled = false;
        if (!componentsInput.value) componentsInput.value = '3';
        iterationsInput.disabled = false;
        if (!iterationsInput.value) iterationsInput.value = '200';
    } else {
        // PCA/UMAP: allow setting output dims only
        componentsInput.disabled = false;
        if (!componentsInput.value) componentsInput.value = '3';
        iterationsInput.disabled = true;
        iterationsInput.value = '';
    }
});

// Initialize component inputs to match default select values on load
document.getElementById('text-dr-model').dispatchEvent(new Event('change'));
document.getElementById('image-dr-model').dispatchEvent(new Event('change'));

// Run loadData once when the script loads (but without blocking the main thread immediately)
// The runAnalysis function will ensure the data is fully loaded before proceeding.
// window.addEventListener('load', () => {
//     // Optionally start loading here, but runAnalysis handles it safely.
// });