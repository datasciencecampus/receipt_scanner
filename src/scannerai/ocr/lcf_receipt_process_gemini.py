"""use Gemini to process an image and output structured information."""

import json
import mimetypes
import os

import cv2
import google.generativeai as genai
import numpy as np
from PIL import Image

from scannerai.utils.scanner_utils import merge_pdf_pages, read_api_key


class LCFReceiptProcessGemini:
    """OCR processor using Gemini."""

    def __init__(self, google_credentials_path=None, gemini_api_key_path=None):
        """Initialize Gemini API with credentials."""

        self.model = None
        self.InitSuccess = False

        if not google_credentials_path or not os.path.exists(
            google_credentials_path
        ):
            print(f"WARNING: Google credentials not found or file does not exist!")
            return

        if not gemini_api_key_path or not os.path.exists(gemini_api_key_path):
            print(f"WARNING: Gemini API key not found or file does not exist!")
            return

        if google_credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
                google_credentials_path
            )

        gemini_api_key = read_api_key(gemini_api_key_path)
        genai.configure(api_key=gemini_api_key)

        self.model = genai.GenerativeModel(
            "gemini-1.5-flash"
        )  # Initialize model here
        self.prompt = """Analyze this receipt image and extract the shop name, items with their prices, total amount, and payment mode. Format the output as a JSON object with the following structure:
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

        self.InitSuccess = True

    def get_InitSuccess(self):
        """Return the initialization status."""
        return self.InitSuccess

    def process_receipt(
        self, file_path, debug_mode=False, enable_price_count=False
    ):  # Add parameters for flexibility
        """Extract structured information from an input image."""

        file_type, _ = mimetypes.guess_type(file_path)

        if file_type == "application/pdf":
            image = merge_pdf_pages(file_path)
        elif file_type and file_type.startswith("image/"):
            image = Image.open(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        if debug_mode:  # Use the passed debug_mode parameter
            opencv_img = np.array(image)
            opencv_img = opencv_img[:, :, ::-1].copy()
            cv2.imshow(f"input image: {file_path}", opencv_img)
            cv2.waitKey(0)

        if enable_price_count:
            print("input image size: ", image.size)

        receipt_data = {
            "shop_name": None,
            "payment_mode": None,
            "total_amount": None,
            "items": [],  # or None, depending on how you want to handle it
            "receipt_pathfile": file_path,
        }

        if self.model:
            response = self.model.generate_content(
                [self.prompt, image]
            )  # Use pre-initialized prompt and model

            if enable_price_count:
                print("token usage:\n", response.usage_metadata)

            receipt_info = response.text
            lpos = receipt_info.find("{")
            receipt_info = receipt_info[lpos:]
            rpos = receipt_info.rfind("}")
            receipt_info = receipt_info[: rpos + 1]

            try:
                receipt_data = json.loads(receipt_info)
            except json.JSONDecodeError:
                return receipt_data  # Or handle the error as needed

        receipt_data["receipt_pathfile"] = file_path

        return receipt_data
