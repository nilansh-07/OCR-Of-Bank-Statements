// Import the llama-ocr module, which provides OCR (Optical Character Recognition) functionality.
const { ocr } = require("llama-ocr");

/**
 * Asynchronously runs the OCR process on a given file using the provided API key.
 * 
 * @param {string} filePath - The path to the file on which OCR will be performed.
 * @param {string} apiKey - The API key required to authenticate with the OCR service.
 * 
 * @throws Will throw an error if filePath or apiKey is missing, or if an issue occurs during OCR processing.
 */
async function runOCR(filePath, apiKey) {
    try {
        // Check if both filePath and apiKey are provided.
        if (!filePath || !apiKey) {
            throw new Error("File path and API key are required arguments.");
        }

        // Perform OCR on the provided file and retrieve the result in markdown format.
        const markdown = await ocr({ filePath, apiKey });

        // Log the OCR result to the console.
        console.log("OCR Result:\n", markdown);
    } catch (error) {
        // Log a user-friendly error message if an error occurs during the OCR process.
        console.error("An error occurred during OCR processing:", error.message);

        // If the error object contains a stack trace, log it for debugging purposes.
        if (error.stack) {
            console.error(error.stack);
        }
    }
}

// Retrieve command-line arguments: the file path (2nd argument) and API key (3rd argument).
const filePath = process.argv[2];
const apiKey = process.argv[3];

// Invoke the OCR function with the provided arguments.
runOCR(filePath, apiKey);
