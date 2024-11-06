"""use Gemini to process an image and output structured information."""

import json
import mimetypes
import os

import cv2
import google.generativeai as genai
import numpy as np
from PIL import Image

from scannerai._config.config import config
from scannerai.utils.scanner_utils import merge_pdf_pages, read_api_key


class LCFReceiptProcessGemini:
    """OCR processor using Gemini."""

    def __init__(self):
        """Initialize Gemini API with credentials."""
        if config.google_credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
                config.google_credentials_path
            )

        if not config.gemini_api_key_path:
            raise ValueError("Gemini API key not found in configuration")

        gemini_api_key = read_api_key(config.gemini_api_key_path)
        genai.configure(api_key=gemini_api_key)

    def process_receipt(self, file_path):
        """Extract structured information from an input image."""

        file_type, _ = mimetypes.guess_type(file_path)

        if file_type == "application/pdf":
            # Convert PDF to a single merged image
            image = merge_pdf_pages(file_path)
        elif file_type and file_type.startswith("image/"):
            # Load the image directly
            image = Image.open(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        if config.debug_mode:
            opencv_img = np.array(image)
            opencv_img = opencv_img[:, :, ::-1].copy()
            cv2.imshow(f"input image: {file_path}", opencv_img)
            cv2.waitKey(0)

        # Use Gemini to analyze the receipt image
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = """Analyze this receipt image and extract the shop name, items with their prices, total amount, and payment mode. Format the output as a JSON object with the following structure:
        {
            "shop_name": "example shop",
            "items": [
                {"name": "item1", "price": 1.99},
                {"name": "item2", "price": 2.49},
                ...
            ],
            "total_amount": 27.83,
            "payment_mode": "card"
        }
        """

        if config.enable_price_count:
            # input_tokens = model.count_tokens([prompt, image])
            print("input image size: ", image.size)
        response = model.generate_content([prompt, image])

        if config.enable_price_count:
            print("token usage:\n", response.usage_metadata)

        receipt_info = response.text
        lpos = receipt_info.find("{")
        receipt_info = receipt_info[lpos:]
        rpos = receipt_info.rfind("}")
        receipt_info = receipt_info[: rpos + 1]

        receipt_data = (
            json.loads(receipt_info)
            if isinstance(receipt_info, str)
            else receipt_info
        )

        receipt_data["receipt_pathfile"] = file_path

        return receipt_data


# Usage example:

# processor = LCFReceiptProcessGemini()
# image_pathfile = os.path.join('/path/to/your/image.jpg')
# result = processor.process_receipt(image_pathfile)
# print(json.dumps(result, indent=2))
