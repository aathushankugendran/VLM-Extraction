# json_parser.py

import json  # Import JSON module for handling structured data
from base_processor import BaseProcessor  # Import the abstract base class
from config import OUTPUT_JSON_PATH  # Import the file path for saving JSON output

class JSONParser(BaseProcessor):
    """
    JSONParser class handles the transformation of extracted text into structured JSON format.
    This ensures that data is properly formatted and saved for further analysis or storage.
    """

    def process(self, extracted_text: str) -> dict:
        """
        Converts extracted text into structured JSON format.

        Args:
            extracted_text (str): The raw text output from the Vision-Language Model.

        Returns:
            dict: Parsed structured data. If parsing fails, returns an error message.
        """
        try:
            return json.loads(extracted_text)  # Attempt to parse the text into a JSON object
        except json.JSONDecodeError:
            # If parsing fails, return an error message
