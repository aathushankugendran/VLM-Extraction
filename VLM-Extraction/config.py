# config.py

import torch  # Import PyTorch for GPU/CPU hardware detection

# ==========================
# 📂 File Paths
# ==========================

# Path to the input PDF file (Change this to your actual file location)
PDF_PATH = "path/to/document.pdf"

# Path where extracted structured JSON data will be saved
OUTPUT_JSON_PATH = "extracted_data.json"

# ==========================
# ⚡ Hardware Configuration
# ==========================

# Automatically detect if a GPU (CUDA) is available, otherwise use CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Prints detected device for debugging (uncomment if needed)
# print(f"Using device: {DEVICE}")

# ==========================
# 🏗️ JSON Schema for Structured Output
# ==========================

# This schema defines the structure of the extracted information from documents.
# The model will return results in this format, ensuring consistency in data storage.
JSON_SCHEMA = {
    "title": "",              # Extracted document title
    "authors": "",            # List of authors (if available)
    "abstract": "",           # Summary of the document
    "keywords": [],           # List of extracted keywords
    "publication_date": "",   # Date of publication (if available)
    "journal": "",            # Journal or source of publication
    "sections": {}            # Dictionary for structured document sections
}

# This JSON structure ensures that every document's extracted details 
# remain organized and easily accessible for further processing.