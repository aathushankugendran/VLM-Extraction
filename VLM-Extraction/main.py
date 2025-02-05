# main.py

# Import necessary components from different modules
from config import PDF_PATH  # File path for the input PDF
from pdf_processor import PDFProcessor  # Converts PDF to an image
from image_processor import ImageProcessor  # Optimizes images for processing
from vlm_model import VLMModel  # Loads and runs the Vision-Language Model
from json_parser import JSONParser  # Parses and saves extracted JSON data

def main():
    """
    Executes