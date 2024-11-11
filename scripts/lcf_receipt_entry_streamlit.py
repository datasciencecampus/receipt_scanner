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

def classify_items(receipt_data, classifier, description_dict):
    """Classify items in receipt data using the LCF classifier."""
    for item in receipt_data["items"]:
        itemDesc = item["name"]
        result, prob = classifier.predict(itemDesc)
        item["code"] = result
        item["prob"] = prob
        
        item["code_desc"] = description_dict.get(str(item["code"]), "")
        
    return receipt_data


def process_image(image_path, ocr_processor, classifier_processor, description_dictionary):
    """Process a single receipt image."""
    # Process receipt using OCR
    receipt_data = ocr_processor.process_receipt(image_path)
    if receipt_data is None:
        return None

    # Classify items
    receipt_data = classify_items(receipt_data, classifier_processor, description_dictionary)

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
                    "code_desc": item["code_desc"],
                    "price": item["price"],
                    "prob": item["prob"],
                    "shop_name": receipt_data["shop_name"],
                    "image_path": receipt_data.get("receipt_pathfile", ""),
                    "payment_mode": receipt_data.get("payment_mode", ""),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(file_path, index=False)


def update_code_descriptions(df, coicop_dict):
    """Update COICOP descriptions based on codes."""
    df = df.copy()
    df['code_desc'] = df['code'].astype(str).map(lambda x: coicop_dict.get(str(x), ""))
    return df


def initialize_session_state():
    """Initialize or reset session state variables."""
    if "results" not in st.session_state:
        print('Initialise st.session_state.results = []')
        st.session_state.results = []
    if "current_index" not in st.session_state:
        print('Initialise st.session_state.current_index = 0')
        st.session_state.current_index = 0
    if "edited_data" not in st.session_state:
        print('Initialise st.session_state.edited_data = {}')
        st.session_state.edited_data = {}
    if "last_edited_df" not in st.session_state:
        print('Initialise st.session_state.last_edited_df = None')
        st.session_state.last_edited_df = None
        
    # initialise OCR processor and text classifier
    # Initialize OCR processor based on config
    if "ocr_processor" not in st.session_state:
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

        st.session_state.ocr_processor = OCRProcessor()
        
    # load text classifier
    if "lcf_classifier" not in st.session_state:
        st.session_state.lcf_classifier = lcf_classifier(
        config.classifier_model_path, config.label_encoder_path)

    # load COICOP description data
    if "coicop_dict" not in st.session_state:
        ROOT_DIR = os.path.abspath(os.curdir)
        coicop_pathfile = os.path.join(ROOT_DIR+'/data/9123_volume_d_expenditure_codes_2021-22.xlsx')
        coicop_df = pd.read_excel(coicop_pathfile, sheet_name='Part 1')
        st.session_state.coicop_dict = dict(zip(coicop_df['LCF CODE'], coicop_df['Description'].str.strip()))


def on_data_change():
    """Callback function for data editor changes."""
    current_index = st.session_state.current_index
    editor_key = f"items_editor_{current_index}"
    
    if editor_key in st.session_state:
        # Get the edit state from the data editor
        edit_state = st.session_state[editor_key]
        
        # Get the current dataframe from our stored state
        current_df = st.session_state.edited_data.get(current_index, pd.DataFrame())
        
        # Handle deleted rows
        if 'deleted_rows' in edit_state and edit_state['deleted_rows']:
            current_df = current_df.drop(edit_state['deleted_rows'])
            # Reset index after deletion to ensure continuous indexing
            current_df = current_df.reset_index(drop=True)
        
        # Apply the edits from edit_state
        if 'edited_rows' in edit_state:
            for idx, row_edits in edit_state['edited_rows'].items():
                if idx < len(current_df):  # Make sure the index exists
                    for col, value in row_edits.items():
                        current_df.at[idx, col] = value
        
        # Handle added rows
        if 'added_rows' in edit_state and edit_state['added_rows']:
            new_rows = pd.DataFrame(edit_state['added_rows'])
            current_df = pd.concat([current_df, new_rows], ignore_index=True)
        
        # Update code descriptions
        current_df = update_code_descriptions(current_df, st.session_state.coicop_dict)
        
        # Store the updated dataframe
        st.session_state.edited_data[current_index] = current_df
        
        # Update the main results with the modified data
        st.session_state.results[current_index]["receipt_data"]["items"] = current_df.to_dict("records")



def main():
    """To execute interface."""
    st.title("Receipt Data Entry System")

    # Initialize session state
    initialize_session_state()

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
                
                #reset results and index
                st.session_state.results = []
                st.session_state.current_index = 0
                st.session_state.edited_data = {}
                st.session_state.last_edited_df = None

                for i, file in enumerate(uploaded_files):
                    # Save temporary file
                    print(file.name)
                    temp_path = f"temp_{file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(file.getvalue())

                    # Process receipt
                    result = process_image(temp_path, st.session_state.ocr_processor,\
                        st.session_state.lcf_classifier, \
                        st.session_state.coicop_dict)
                    
                    if result:
                        st.session_state.results.append(result)

                    # Update progress
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    os.remove(temp_path)
                    
                    # print('st.session_state.results:\n', st.session_state.results)

                st.success(
                    f"Processed {len(st.session_state.results)} receipts"
                )
                

        # Navigation with state preservation
        if st.session_state.results:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Previous") and st.session_state.current_index > 0:
                    # Save current state before navigation
                    st.session_state.current_index -= 1
            with col2:
                if st.button("Next") and st.session_state.current_index < len(st.session_state.results) - 1:
                    # Save current state before navigation
                    st.session_state.current_index += 1
            st.write(f"Receipt {st.session_state.current_index + 1} of {len(st.session_state.results)}")

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
        current_result = st.session_state.results[st.session_state.current_index]
        current_index = st.session_state.current_index

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
            
            # Create unique keys for each input field
            shop_key = f"shop_name_{current_index}"
            total_key = f"total_amount_{current_index}"
            payment_key = f"payment_mode_{current_index}"
            
            new_shop_name = st.text_input(
                "Shop Name",
                value=receipt_data["shop_name"],
                key=shop_key
            )
            new_total = st.text_input(
                "Total Amount",
                value=receipt_data.get("total_amount", ""),
                key=total_key
            )
            new_payment_mode = st.text_input(
                "Payment Mode",
                value=receipt_data.get("payment_mode", ""),
                key=payment_key
            )

            # Update values in session state
            receipt_data["shop_name"] = new_shop_name
            receipt_data["total_amount"] = new_total
            receipt_data["payment_mode"] = new_payment_mode

            # Items table
            st.subheader("Items")
            
            # Get the current edited data from session state or create new
            editor_key = f"items_editor_{current_index}"
            
             # Initialize or get the current dataframe
            if current_index not in st.session_state.edited_data:
                items_df = pd.DataFrame(current_result["receipt_data"]["items"])
                items_df = update_code_descriptions(items_df, st.session_state.coicop_dict)
                st.session_state.edited_data[current_index] = items_df
            
            items_df = st.session_state.edited_data[current_index]
            
            # Display the data editor
            _ = st.data_editor(
                items_df,
                num_rows="dynamic",
                key=editor_key,
                column_config={
                    "name": "Item Name",
                    "price": "Price",
                    "code": "COICOP Code",
                    "code_desc": st.column_config.Column(
                        "COICOP Description",
                        disabled=True,
                    ),
                    "prob": st.column_config.NumberColumn(
                        "Confidence Score",
                        format="%.2f",
                    ),
                },
                on_change=on_data_change
            )
            
            # Add a "Save Changes" button to force update
            # if st.button("Save Changes", key=f"save_changes_{current_index}"):
            #     st.success("Changes saved successfully!")
            #     st.experimental_rerun()
                    
    else:
        st.info("Upload receipt images to begin processing")
        
        
if __name__ == "__main__":
    main()
