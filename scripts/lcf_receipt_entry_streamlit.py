"""use streamlit to create interface of receipt data entry."""

import json
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from scannerai._config.config import config
from scannerai.classifiers.lcf_classify import lcf_classifier
from scannerai.utils.scanner_utils import merge_pdf_pages

# Configure Streamlit page
st.set_page_config(
    layout="wide",
    page_title="Living Costs and Food Survey - Receipt Data Entry",
)

# Initialize OCR processor based on config
if config.ocr_model == 1:
    from scannerai.ocr.lcf_receipt_process_openai import (
        LCFReceiptProcessOpenai as OCRProcessor,
    )

    st.sidebar.info("Using OpenAI OCR Model")
elif config.ocr_model == 2:
    from scannerai.ocr.lcf_receipt_process_gpt4vision import (
        LCFReceiptProcessGPT4Vision as OCRProcessor,
    )

    st.sidebar.info("Using GPT-4 Vision OCR Model")
elif config.ocr_model == 3:
    from scannerai.ocr.lcf_receipt_process_gemini import (
        LCFReceiptProcessGemini as OCRProcessor,
    )

    st.sidebar.info("Using Gemini OCR Model")
else:
    st.error("Error: No OCR Model is set!")
    st.stop()

ocr_processor = OCRProcessor()
lcf_classifier = lcf_classifier(
    config.classifier_model_path, config.label_encoder_path
)


def classify_items(receipt_data):
    """Classify items in receipt data using the LCF classifier."""
    for item in receipt_data["items"]:
        itemDesc = item["name"]
        result, prob = lcf_classifier.predict(itemDesc)
        item["code"] = result
        item["prob"] = prob
    return receipt_data


def process_image(image_path):
    """Process a single receipt image."""
    # Process receipt using OCR
    receipt_data = ocr_processor.process_receipt(image_path)
    if receipt_data is None:
        return None

    # Classify items
    receipt_data = classify_items(receipt_data)

    # Read the image
    if image_path.lower().endswith((".png", ".jpg", ".jpeg")):
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    elif image_path.lower().endswith(".pdf"):
        original_image = merge_pdf_pages(image_path)
        original_image = np.array(original_image)

    return {"image": original_image, "receipt_data": receipt_data}


def save_to_json(results, file_path):
    """Save results to JSON file."""
    serializable_results = [
        {"receipt_data": result["receipt_data"]} for result in results
    ]
    with open(file_path, "w") as json_file:
        json.dump(serializable_results, json_file, indent=4)


def save_to_csv(results, file_path):
    """Save results to CSV file."""
    rows = []
    for result in results:
        receipt_data = result["receipt_data"]
        for item in receipt_data["items"]:
            rows.append(
                {
                    "item": item["name"],
                    "code": item["code"],
                    "price": item["price"],
                    "prob": item["prob"],
                    "shop_name": receipt_data["shop_name"],
                    "image_path": receipt_data.get("receipt_pathfile", ""),
                    "payment_mode": receipt_data.get("payment_mode", ""),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(file_path, index=False)


def main():
    """To execute interface."""
    st.title("Receipt Data Entry System")

    # Initialize session state if not exists
    if "results" not in st.session_state:
        st.session_state.results = []
        st.session_state.current_index = 0

    # Sidebar for file upload and navigation
    with st.sidebar:
        st.header("Upload & Navigation")

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload receipt images",
            type=["png", "jpg", "jpeg", "pdf"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            if st.button("Process Uploaded Files"):
                progress_bar = st.progress(0)
                st.session_state.results = []

                for i, file in enumerate(uploaded_files):
                    # Save temporary file
                    temp_path = f"temp_{file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(file.getvalue())

                    # Process receipt
                    result = process_image(temp_path)
                    if result:
                        st.session_state.results.append(result)

                    # Update progress
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    os.remove(temp_path)

                st.success(
                    f"Processed {len(st.session_state.results)} receipts"
                )
                st.session_state.current_index = 0

        # Navigation
        if st.session_state.results:
            st.write(
                f"Receipt {st.session_state.current_index + 1} of {len(st.session_state.results)}"
            )
            col1, col2 = st.columns(2)
            with col1:
                if (
                    st.button("Previous")
                    and st.session_state.current_index > 0
                ):
                    st.session_state.current_index -= 1
            with col2:
                if (
                    st.button("Next")
                    and st.session_state.current_index
                    < len(st.session_state.results) - 1
                ):
                    st.session_state.current_index += 1

        # Export options
        if st.session_state.results:
            st.header("Export Data")
            export_format = st.selectbox("Export format", ["JSON", "CSV"])
            if st.button("Export"):
                if export_format == "JSON":
                    save_to_json(st.session_state.results, "receipt_data.json")
                    st.success("Data exported to receipt_data.json")
                else:
                    save_to_csv(st.session_state.results, "receipt_data.csv")
                    st.success("Data exported to receipt_data.csv")

    # Main content area
    if st.session_state.results:
        current_result = st.session_state.results[
            st.session_state.current_index
        ]

        # Display receipt image and data side by side
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Receipt Image")
            image = current_result["image"]
            # Draw bounding boxes
            image_with_boxes = image.copy()
            for item in current_result["receipt_data"]["items"]:
                if "bounding_boxes" in item and item["bounding_boxes"]:
                    for x, y, w, h in item["bounding_boxes"]:
                        cv2.rectangle(
                            image_with_boxes,
                            (x, y),
                            (x + w, y + h),
                            (0, 255, 0),
                            2,
                        )
            st.image(image_with_boxes, use_column_width=True)

        with col2:
            st.subheader("Receipt Data")

            # Shop details
            receipt_data = current_result["receipt_data"]
            new_shop_name = st.text_input(
                "Shop Name", value=receipt_data["shop_name"]
            )
            new_total = st.text_input(
                "Total Amount", value=receipt_data.get("total_amount", "")
            )
            new_payment_mode = st.text_input(
                "Payment Mode", value=receipt_data.get("payment_mode", "")
            )

            # Update values
            receipt_data["shop_name"] = new_shop_name
            receipt_data["total_amount"] = new_total
            receipt_data["payment_mode"] = new_payment_mode

            # Items table
            st.subheader("Items")
            items_df = pd.DataFrame(receipt_data["items"])
            edited_df = st.data_editor(
                items_df,
                num_rows="dynamic",
                column_config={
                    "name": "Item Name",
                    "price": "Price",
                    "code": "COICOP Code",
                    "prob": "Confidence Score",
                },
            )

            # Update items in receipt data
            receipt_data["items"] = edited_df.to_dict("records")
    else:
        st.info("Upload receipt images to begin processing")


if __name__ == "__main__":
    main()
