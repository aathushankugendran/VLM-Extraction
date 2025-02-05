# pdf_processor.py

# Import required libraries
from pdf2image import convert_from_path  # Converts PDF pages into images
from PIL import Image  # Handles image processing
from base_processor import BaseProcessor  # Inherit structure from BaseProcessor class

class PDFProcessor(BaseProcessor):
    """
    PDFProcessor class extracts the first page of a PDF and converts it into an image.
    This image is later used for processing by the Vision-Language Model.
    """

    def process(self, pdf_path: str) -> Image:
        """
        Extracts the first page of a PDF and converts it into an image.

        Args:
            pdf_path (str): Path to the PDF file.