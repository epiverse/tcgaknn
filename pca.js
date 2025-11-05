// pca.js

/**
 * Global variable to hold the Pyodide instance.
 * It is initialized once when the script loads.
 */
let pyodideInstance = null;
let pyodideInitializationPromise = null;

// --- Pyodide Initialization ---

/**
 * Function to initialize Pyodide and load the scikit-learn package.
 * It ensures initialization happens only once.
 */
async function initializePyodide() {
    if (pyodideInstance) {
        return pyodideInstance;
    }

    if (pyodideInitializationPromise) {
        return pyodideInitializationPromise;
    }

    if (typeof loadPyodide === "undefined") {
        console.error("Pyodide is not loaded. Ensure the pyodide.js script is included in your HTML.");
        throw new Error("Pyodide is not loaded.");
    }

    pyodideInitializationPromise = (async () => {
        console.log("Initializing Pyodide for PCA...");
        const pyodide = await loadPyodide({
            indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.3/full/",
        });
        await pyodide.loadPackage("scikit-learn");
        pyodideInstance = pyodide;
        console.log("Pyodide initialized successfully and scikit-learn loaded.");
        return pyodideInstance;
    })();

    return pyodideInitializationPromise;
}

// Immediately start the initialization when the script is loaded
// Do not eagerly initialize Pyodide on script load; initialize lazily when PCA is requested.
// initializePyodide();


// --- Core PCA Function ---

/**
 * Performs PCA transformation on the given embeddings to the specified number of components
 * using the initialized Pyodide environment.
 * @param {number[][]} embeddings - The array of vectors (data points) to transform.
 * @param {number} components - The desired number of dimensions (n_components).
 * @returns {Promise<number[][]>} A promise that resolves to the dimensionally-reduced embeddings.
 */
async function pcaTransform(embeddings, components) {
    const pyodide = await initializePyodide(); // Wait for initialization if needed

    // Check if the input is valid before sending to Python
    if (!Array.isArray(embeddings) || embeddings.length === 0 || components < 1) {
        console.warn("Invalid embeddings or component count for PCA.");
        return embeddings.map(e => e.slice(0, components)); // Fallback
    }

    // Convert JavaScript array to a JSON string for efficient transfer
    const jsonData = JSON.stringify(embeddings);
    pyodide.globals.set("X_json", jsonData);
    pyodide.globals.set("n_components", components);

    try {
        const transformedData = pyodide.runPython(`
            import json
            import numpy as np
            from sklearn.decomposition import PCA

            # Load data from the JavaScript environment
            X = np.array(json.loads(X_json))
            if X.ndim != 2:
                # Handle non-2D input gracefully
                print('Input data must be a 2D array. Returning original.')
                X.tolist() # Fallback to original data structure

            del X_json

            # Perform PCA
            pca = PCA(n_components=n_components)
            # fit_transform will automatically center the data
            X_transformed = pca.fit_transform(X).tolist()

            X_transformed
        `);
        // Convert the Pyodide result back to a standard JavaScript array
        return transformedData.toJs();

    } catch (error) {
        console.error("Error during Pyodide PCA transformation:", error);
        // Fallback to slicing the original data in case of error
        return embeddings.map(e => e.slice(0, components));
    }
}


// --- Handler for analysis.js ---

/**
 * The unified function signature expected by analysis.js for Dimensionality Reduction.
 * This function handles the "no_model" case and calls the asynchronous PCA logic.
 * @param {number[][]} embeddings - The input embeddings.
 * @param {string} model - The model type (must be 'pca' here).
 * @param {number} components - The number of components.
 * @returns {Promise<number[][]>} The reduced embeddings.
 */
function applyPCA(embeddings, components) {
    // Note: Since this is an external helper, we rely on analysis.js to handle
    // the 'no_model' check and the component logic.
    // We return a Promise because pcaTransform is asynchronous.
    return pcaTransform(embeddings, components);
}