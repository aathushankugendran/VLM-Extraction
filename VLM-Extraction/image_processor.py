# image_processor.py

from PIL import Image  # Import PIL for image processing
from base_processor import BaseProcessor  # Import the abstract base class for structure

class ImageProcessor(BaseProcessor):
    """
    ImageProcessor class handles image preprocessing for the Vision-Language Model.
    This includes ensuring images are in the correct format and resizing them 
    while maintaining aspect ratio to optimize model performance.
    """

    def process(self, image: Image) -> Image:
        """
        Converts an image to RGB format (if needed) and resizes it while maintaining aspect ratio.

        Args:
            image (PIL.Image): The extracted image from the PDF.

        Returns:
            Image: Optimized image ready for processing by the Vision-Language Model.
        """
        
        # ==========================
        # 🖼️ Ensure Image is in RGB Format
        # ==========================
        if image.mode != "RGB":
            imag