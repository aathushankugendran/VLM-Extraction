# vlm_model.py

# Import necessary libraries
import torch  # PyTorch for deep learning model execution
import json  # JSON for structuring extracted information
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor  # Model & processor for VLM
from base_processor import BaseProcessor  # Import base processor for structure
from config import DEVICE, JSON_SCHEMA  # Import configuration constants (hardware & JSON schema)
from PIL import Image  # Import Image module for type hinting

class VLMModel(BaseProcessor):
    """
    VLMModel class handles inference using the Qwen2-VL Vision-Language Model.
    It processes an image and extracts structured textual data in JSON format.
    """

    def __init__(self):
        """
        Initializes the model and processor, setting up GPU acceleration for efficiency.
        """
        # Load the processor for handling text and images
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

        # Load the pre-trained Qwen2-VL model with optimized settings for Databricks
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch.bfloat16,  # Uses bfloat16 for memory-efficient inference
            device_map="auto"  # Automatically assigns model to the best available device (CPU/GPU)
        ).to(DEVICE)  # Move the model to the detected hardware (GPU if available)

        # Compile the model for performance optimization (only if GPU is available)
        if torch.cuda.is_available():
            self.model = torch.compile(self.model)  # Reduces execution time via graph optimizations

    def process(self, image: Image) -> str:
        """
        Processes an image using the Vision-Language Model (VLM) to extract structured text.

        Args:
            image (PIL.Image): The processed image to be analyzed.

        Returns:
            str: Extracted text formatted as structured JSON.
        """
        
        # ==========================
        # 📝 Define User Prompt for Model
        # ==========================
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},  # Attach the optimized image
                    {"type": "text", "text": f"""
                        Extract structured information in JSON format:
                        {json.dumps(JSON_SCHEMA, indent=4)}
                    """}  # Provide a clear schema to ensure consistent output
                ]
            }
        ]

        # ==========================
        # 📥 Prepare Inputs for Model
        # ==========================
        inputs = self.processor(
            text=self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            images=image,  # Provide the processed image
            padding=True,  # Ensure proper input padding
            return_tensors="pt"  # Convert to PyTorch tensors for model inference
        ).to(DEVICE)  # Move tensors to the appropriate device (GPU/CPU)

        # ==========================
        # 🚀 Run Model & Generate Output
        # ==========================
        output = self.model.generate(**inputs, max_new_tokens=500)  # Limit response to 500 tokens

        # ==========================
        # 📤 Convert Output to Readable Text
        # ==========================
        return self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
