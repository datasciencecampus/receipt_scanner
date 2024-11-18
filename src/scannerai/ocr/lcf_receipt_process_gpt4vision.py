"""Use GPT4 vision API to extract structured information from an image."""

import base64
import io
import json
import os

from openai import OpenAI
from PIL import Image

from scannerai.utils.scanner_utils import (
    count_tokens_openai,
    estimate_image_tokens_openai,
    merge_pdf_pages,
    read_api_key,
)


class LCFReceiptProcessGPT4Vision:
    """Class to extract text from image using OpenAI ChatGPT4 vision API."""

    def __init__(self, openai_api_key_path):
        """Initialize Openai API with credentials."""

        self.InitSuccess = False  # Initialize to False

        self.client = None

        if not openai_api_key_path or not os.path.exists(openai_api_key_path):
            print(
                f"WARNING: ChatGPT API key not found or file does not exist: {openai_api_key_path}"
            )
            return

        openai_api_key = read_api_key(openai_api_key_path)
        # Initialize OpenAI client
        self.client = OpenAI(api_key=openai_api_key)

        # prompt is same for calling, so initialised
        self.prompt = (
            "Please extract the following information from this receipt image and output it in the form of a json dictionary:\n"
            "if any value is missing, please leave it empty\n"
            "Do not allow item name empty\n"
            "{\n"
            "    'shop_name': 'example shop',\n"
            "    'items': [\n"
            "        {'name': 'item1', 'price': 1.99},\n"
            "        {'name': 'item2', 'price': 2.49},\n"
            "        ...\n"
            "    ],\n"
            "    'total_amount': 27.83,\n"
            "    'payment_mode': 'card'\n"
            "}"
        )
        self.InitSuccess = True

    def get_InitSuccess(self):
        """Return the initialization status."""
        return self.InitSuccess

    @staticmethod
    def encode_image(image):
        """Encode an image to the required format of input for gpt4 vision API."""

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def estimate_vision_tokens(self, messages, image_width, image_height):
        """Estimate vision api tokens."""
        token_count = 0
        token_text = 0
        token_image = 0
        for message in messages:
            if isinstance(message["content"], list):
                for content in message["content"]:
                    if content["type"] == "text":
                        token_text += count_tokens_openai(
                            "gpt-4o-mini", content["text"]
                        )
                        token_count += count_tokens_openai(
                            "gpt-4o-mini", content["text"]
                        )
                    elif content["type"] == "image_url":
                        token_image += estimate_image_tokens_openai(
                            image_width, image_height
                        )
                        token_count += estimate_image_tokens_openai(
                            image_width, image_height
                        )
            else:
                token_count += count_tokens_openai(
                    "gpt-4o-mini", message["content"]
                )
        return token_count, token_text, token_image

    def process_receipt(self, image_path, enable_price_count=False):
        """Process receipt."""
        file_extension = os.path.splitext(image_path)[1].lower()
        if file_extension in [".jpg", ".jpeg", ".png"]:
            with Image.open(image_path) as image:
                base64_image = LCFReceiptProcessGPT4Vision.encode_image(image)
        elif file_extension == ".pdf":
            image = merge_pdf_pages(image_path)

            if image:
                base64_image = LCFReceiptProcessGPT4Vision.encode_image(image)
            else:
                raise ValueError("Failed to convert PDF to image")
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        receipt_data = {
            "shop_name": None,
            "payment_mode": None,
            "items": [],  # or None, depending on how you want to handle it
            "receipt_pathfile": image_path,
        }

        if self.get_InitSuccess():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ]

            max_tokens = 1000
            response = self.client.chat.completions.create(
                model="gpt-4o-mini", messages=messages, max_tokens=max_tokens
            )
            receipt_info = response.choices[0].message.content

            if response.choices[0].finish_reason == "length":
                print(
                    "ERROR: THE RESULTS EXCEEDED the max_token, so be partically cut off!"
                )
            else:
                lpos = 0
                lpos = receipt_info.find("{")
                receipt_info = receipt_info[lpos:]
                rpos = receipt_info.rfind("}")
                receipt_info = receipt_info[: rpos + 1]

                receipt_data = (
                    json.loads(receipt_info)
                    if isinstance(receipt_info, str)
                    else receipt_info
                )

        receipt_data["receipt_pathfile"] = image_path

        if enable_price_count:
            width, height = image.size
            input_tokens, input_tokens_text, input_tokens_image = (
                self.estimate_vision_tokens(messages, width, height)
            )

            output_tokens = count_tokens_openai("gpt-4o-mini", receipt_info)
            total_tokens = input_tokens + output_tokens
            print(
                f"Estimated input tokens: {input_tokens}, text tokens: {input_tokens_text}, image tokens: {input_tokens_image}"
            )
            print(f"Output tokens: {output_tokens}")
            print(f"Estimated total tokens: {total_tokens}")

        return receipt_data


# Example usage
# image_pathfile = '/path/to/your/image.jpg'
# processor = LCFReceiptProcessGPT4Vision()
# receipt_data = processor.process_receipt(image_pathfile)
# if receipt_data is not None:
#     print(json.dumps(receipt_data, indent=2))
