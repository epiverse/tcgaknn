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
    setStatus('Loading data...');
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

        const tsvContent = await file.async('string');
        console.log("Successfully unzipped and loaded TSV content. Parsing data...");

        // 3. Parse the TSV content
        const rows = tsvContent.trim().split('\n');
        // Headers are typically the first line, split by tab. Use 'embeddings' for text.
        const headers = rows[0].split('\t');
        const data = [];

        for (let i = 1; i < rows.length; i++) {
            const values = rows[i].split('\t');
            const row = {};
            headers.forEach((header, index) => {
                // Remove carriage returns often present in TSV files
                row[header.trim()] = values[index] ? values[index].trim() : '';
            });

            // Map the generic 'embeddings' column to 'text_embedding'
            // and the specific 'image_embedding' column.
            data.push({
                i: parseInt(row.i),
                id: row.id,
                patient_id: row.patient_id,
                cancer_type: row.cancer_type,
                // Use 'embeddings' for the Text KNN analysis
                text_embedding: parseEmbedding(row.embeddings),
                // Use 'image_embedding' for the Image KNN analysis
                image_embedding: parseEmbedding(row.image_embedding)
            });
        }
    console.log(`Data parsing complete. Loaded ${data.length} samples.`);
    setStatus('Data loaded');
        return data.filter(d => d.text_embedding.length > 0 && d.image_embedding.length > 0);

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
async function applyDR(embeddings, model, components) {
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
        // Pass the component value (which is iterations for t-SNE)
        return await applyTSNE(embeddings, components);
    }

    return embeddings;
}

// --- 4. K-NEAREST NEIGHBORS (KNN) ---

// This function is ASYNCHRONOUS because it calls applyDR
// NOTE: We now include the components parameter explicitly to pass the correct value to applyDR
async function getKNN(data, K, embeddingKey, drModel, components, metric) {
    const distanceFn = getDistanceFunction(metric);
    const rawEmbeddings = data.map(d => d[embeddingKey]);

    if (embeddingKey === 'text_embedding') setStatus('Calculating KNN for text...');
    if (embeddingKey === 'image_embedding') setStatus('Calculating KNN for image...');

    // 1. Apply DR - Use AWAIT here, passing the correct components value
    const processedEmbeddings = await applyDR(rawEmbeddings, drModel, components);

    // 2. Calculate Distances and Find K Neighbors
    const knnResults = {}; // Map: id -> [neighbor_id1, neighbor_id2, ...]

    for (let i = 0; i < processedEmbeddings.length; i++) {
        const queryId = data[i].id;
        const queryEmb = processedEmbeddings[i];

        const distances = [];
        for (let j = 0; j < processedEmbeddings.length; j++) {
            const candidateId = data[j].id;
            if (queryId !== candidateId) { // Exclude self
                const dist = distanceFn(queryEmb, processedEmbeddings[j]);
                distances.push({ id: candidateId, dist: dist });
            }
        }

        // Sort by distance and take the top K
        distances.sort((a, b) => a.dist - b.dist);
        const neighbors = distances.slice(0, K).map(d => d.id);
        knnResults[queryId] = neighbors;
    }
    return knnResults;
}

// --- 5. MAIN ANALYSIS FUNCTION ---

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
        const textComponents = parseInt(document.getElementById('text-components').value);

        const imageDrModel = document.getElementById('image-dr-model').value;
        const imageMetric = document.getElementById('image-metric').value;
        const imageComponents = parseInt(document.getElementById('image-components').value);

        if (isNaN(K) || K <= 0) {
            alert("K (Nearest Neighbors) must be a positive number.");
            return;
        }

    document.getElementById('knn-results').innerHTML = `<p>Calculating KNN and Overlap...</p>`;
    setStatus('Running models and calculating KNN...');


        // --- A. Calculate KNN for Text ---
        // Changed getKNN signature to pass explicit components value
    const textKNN = await getKNN(data, K, 'text_embedding', textDrModel, textComponents, textMetric);

        // --- B. Calculate KNN for Image ---
        // Changed getKNN signature to pass explicit components value
    const imageKNN = await getKNN(data, K, 'image_embedding', imageDrModel, imageComponents, imageMetric);

    // --- C. Calculate Overlap per Sample ---
    setStatus('Calculating overlap between text and image KNN...');
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
                    Total Samples: ${counts.length}
                </div>
            `;
        }

        const overallAverageOverlap = totalOverlapSum / totalPatientSamples;

    // --- E. Display Results ---
    setStatus('Done');
        document.getElementById('knn-results').innerHTML = htmlOutput;
        document.getElementById('average-overlap').innerHTML = `
            Overall Average Overlap of Nearest Neighbors (K=${K}): ${overallAverageOverlap.toFixed(4)}
        `;

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
    const model = this.value;

    if (model === 'tsne') {
        componentsInput.disabled = false; // Enable for t-SNE (iterations)
        componentsInput.value = '200';    // Default iterations for t-SNE
    } else {
        // PCA, UMAP, or No Model
        componentsInput.disabled = false; // Keep enabled for PCA/UMAP (target dimension)
        if (model === 'no_model') {
             componentsInput.disabled = true;
        }
        componentsInput.value = '3';      // Default component count for PCA/UMAP
    }
});

document.getElementById('image-dr-model').addEventListener('change', function() {
    const componentsInput = document.getElementById('image-components');
    const model = this.value;

    if (model === 'tsne') {
        componentsInput.disabled = false; // Enable for t-SNE (iterations)
        componentsInput.value = '200';    // Default iterations for t-SNE
    } else {
        // PCA, UMAP, or No Model
        componentsInput.disabled = false; // Keep enabled for PCA/UMAP (target dimension)
        if (model === 'no_model') {
             componentsInput.disabled = true;
        }
        componentsInput.value = '3';      // Default component count for PCA/UMAP
    }
});

// Run loadData once when the script loads (but without blocking the main thread immediately)
// The runAnalysis function will ensure the data is fully loaded before proceeding.
// window.addEventListener('load', () => {
//     // Optionally start loading here, but runAnalysis handles it safely.
// });