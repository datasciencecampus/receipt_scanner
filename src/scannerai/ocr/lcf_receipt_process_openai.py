"""use Tesseract + GPT-3 to extract structured information from an image."""

import ast
import os
import re

import cv2
import numpy as np
import pandas as pd
import pytesseract
from openai import OpenAI

from scannerai.utils.scanner_utils import (
    count_tokens_openai,
    merge_pdf_pages,
    read_api_key,
)


class LCFReceiptProcessOpenai:
    """class to extract text from image using OpenAI API."""

    def __init__(self, openai_api_key_path):
        """Initialize Openai API with credentials."""

        self.InitSuccess = False  # Initialize to False
        self.client = None  # Initialize to None

        if not openai_api_key_path or not os.path.exists(openai_api_key_path):
            print(
                f"WARNING: ChatGPT API key not found or file does not exist: {openai_api_key_path}"
            )
            return

        openai_api_key = read_api_key(openai_api_key_path)
        # Initialize OpenAI client
        self.client = OpenAI(api_key=openai_api_key)
        self.InitSuccess = True

    def get_InitSuccess(self):
        """Return the initialization status."""
        return self.InitSuccess

    def estimate_tokens(self, messages):
        """Estimate token counts."""
        token_count = 0
        token_text = ""
        for message in messages:
            if isinstance(message["content"], list):
                for content in message["content"]:
                    token_text += count_tokens_openai(
                        "gpt-3.5-turbo", content["text"]
                    )
            else:
                token_count += count_tokens_openai(
                    "gpt-3.5-turbo", message["content"]
                )
        return token_count

    # Function to call OpenAI API and format the receipt information
    def extract_receipt_with_chatgpt(self, ocr_text, enable_price_count=False):
        """To call OpenAI API and format the receipt information."""
        prompt = (
            "Here is the OCR text from a receipt:\n"
            f"'''{ocr_text}'''\n"
            "Please extract the following information and output it in the form of a json dictionary:\n"
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

        # try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",  # Changed from GPT-4 to GPT-3.5-turbo
            messages=messages,
            temperature=0.5,
        )

        # Extract the response text
        # Print the response
        receipt_info = response.choices[0].message.content

        if enable_price_count:
            input_tokens = self.estimate_tokens(messages)
            output_tokens = count_tokens_openai("gpt-3.5-turbo", receipt_info)
            total_tokens = input_tokens + output_tokens
            print(f"Estimated input tokens: {input_tokens}")
            print(f"Output tokens: {output_tokens}")
            print(f"Estimated total tokens: {total_tokens}")

        return receipt_info

    def parse_receipt_info(self, receipt_info):
        """Parse the input text to structured data."""
        receipt_data = {
            "shop_name": None,
            "items": [],
            "total": None,
            # "payment_mode": None,
            # "date": None,
            # "time": None,
        }

        shop_name = ""
        items_with_prices = []
        total_amount = ""
        payment_mode = ""

        lines = receipt_info.split("\n")
        for line in lines:
            line = line.replace("*", "")
            if re.search(
                "shop name", line, re.IGNORECASE
            ):  # line.startswith("1. Shop Name:"):
                shop_name = line.split(":")[1].strip()
            elif re.search(
                "list of items with their prices", line, re.IGNORECASE
            ):  # line.startswith("2. List of items with their prices:"):
                items_start = lines.index(line) + 1
                for item_line in lines[items_start:]:
                    if item_line.startswith("3."):
                        break
                    items_with_prices.append(item_line.strip())
            elif re.search(
                "Total amount paid", line, re.IGNORECASE
            ):  # line.startswith("3. Total amount paid:"):
                total_amount = line.split(":")[1].strip()
            elif re.search(
                "Payment mode", line, re.IGNORECASE
            ):  # line.startswith("4. Payment mode:"):
                payment_mode = line.split(":")[1].strip()

        receipt_data["shop_name"] = shop_name

        # extract item description and price
        item_regex = re.compile(r"\s*-\s*(.*?)\s*(?::\s*£(\d+\.\d{2}))?$")

        # List to hold dictionaries of each parsed item
        for item_price in items_with_prices:
            item_match = item_regex.match(item_price)
            item_description = None
            if item_match:
                item_description = item_match.group(1).strip()
                item_price = item_match.group(2)
                if item_price:
                    item_price = int(float(item_match.group(2).strip()) * 100)
                else:
                    item_price = None  # no price provided
            receipt_data["items"].append(
                {
                    "name": item_description,
                    "price": item_price,
                    "bounding_boxes": None,
                }
            )

        receipt_data["total"] = {"total": total_amount, "bounding_boxes": None}

        receipt_data["payment_mode"] = payment_mode

        return receipt_data

    def format_receipt_info(self, input_dict):
        """Convert dictionary style receipt data to the testing data style."""
        outputs_df = pd.DataFrame(
            columns=["image_relative_path", "shop_name", "recdesc", "amtpaid"]
        )

        # List to hold dictionaries of each parsed item
        rows = []
        for item in input_dict["items"]:
            # Create a dictionary for each row (item)
            row = {
                "image_relative_path": input_dict["receipt_pathfile"],
                "shop_name": input_dict["shop_name"],
                "recdesc": item["name"],
                "amtpaid": item["price"],
            }
            rows.append(row)

        # Use pd.concat to append all rows to the DataFrame at once
        if rows:
            outputs_df = pd.concat(
                [outputs_df, pd.DataFrame(rows)], ignore_index=True
            )

        return outputs_df

    def process_receipt(self, image_path, enable_price_count=False):
        """To extract structured data from input image."""
        # Load image
        file_extension = os.path.splitext(image_path)[1].lower()
        if file_extension in [".jpg", ".jpeg", ".png"]:
            image = cv2.imread(image_path)
        elif file_extension == ".pdf":
            image = merge_pdf_pages(image_path)

            if image:
                # convert to cv2 format
                image = np.array(image)
                image = image[:, :, ::-1].copy()
            else:
                raise ValueError("Failed to convert PDF to image")
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        if image is None:
            print(f"Error: Unable to read image at {image_path}")
            return None

        receipt_data = {
            "shop_name": None,
            "payment_mode": None,
            "items": [],  # or None, depending on how you want to handle it
            "receipt_pathfile": image_path,
        }

        if not self.get_InitSuccess():
            return receipt_data

        # pre-processing image
        if enable_price_count:
            # processed_image = preprocess_image(image)
            # TO ADD...
            processed_image = image
        else:
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ocr_text = pytesseract.image_to_string(processed_image)

        # Call the function
        receipt_info = self.extract_receipt_with_chatgpt(
            ocr_text, enable_price_count
        )
        print(receipt_info)
        # Parse the receipt_info
        # receipt_data = parse_receipt_info(receipt_info)
        receipt_data = (
            ast.literal_eval(receipt_info)
            if isinstance(receipt_info, str)
            else receipt_info
        )

        receipt_data["receipt_pathfile"] = image_path

        return receipt_data


# Example usage
# image_pathfile = '/path/to/your/image.jpg'
# processor = LCFReceiptProcessOpenai()
# receipt_data = processor.process_receipt(image_pathfile)
# if receipt_data is not None:
#     print(receipt_data)
