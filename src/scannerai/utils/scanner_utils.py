"""to store shared utils functions."""

import pdf2image
import tiktoken  # Import tiktoken for token counting
from PIL import Image


# Read API key file
def read_api_key(file_path):
    """Read api key."""
    try:
        with open(file_path, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except IOError:
        print(f"Error: Could not read the file '{file_path}'.")
        return None


# merge multiple pages of a pdf file to one image
def merge_pdf_pages(pdf_path):
    """Convert PDF to images."""
    images = pdf2image.convert_from_path(pdf_path, dpi=300)

    # Calculate the total height of all images
    total_width = max(image.width for image in images)
    total_height = sum(image.height for image in images)

    # Create a new image with the total height
    merged_image = Image.new("RGB", (total_width, total_height))

    # Paste all images into the new image
    y_offset = 0
    for image in images:
        merged_image.paste(image, (0, y_offset))
        y_offset += image.height

    return merged_image


# openai count token count for text
def count_tokens_openai(model, text):
    """Count token for text."""
    encoding = tiktoken.encoding_for_model(model)  # ("gpt-4")
    return len(encoding.encode(text))


# openai, count token count for image
def estimate_image_tokens_openai(width, height):
    """Estimate cost for including an image."""
    tokens = 85

    # Determine the number of 512px tiles
    tiles_x = (width + 511) // 512
    tiles_y = (height + 511) // 512
    total_tiles = tiles_x * tiles_y

    # Add tokens based on the number of tiles
    tokens += total_tiles * 170

    return tokens
