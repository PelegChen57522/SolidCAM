# MistralAI_OCR.py (Adding include_image_base64=True)

import os
import logging
import json
import time
import mimetypes
from mistralai import Mistral
from dotenv import load_dotenv
import sys

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Input/Output Paths ---
PDF_FILE_PATH = "/Users/pelegchen/SolidCAM_ChatBotImageEmbeddings/pdf_files/Milling 2024 Machining Processes.pdf"
OUTPUT_OCR_JSON_FILE = "processed_solidcam_doc.json"

# --- MistralAI Configuration ---
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    logging.error("MISTRAL_API_KEY not found")
    sys.exit("Error: Missing MISTRAL_API_KEY.")
MISTRAL_OCR_MODEL = "mistral-ocr-latest"
FILE_UPLOAD_PURPOSE = 'ocr'

def run_ocr_on_pdf(pdf_path: str, output_json_path: str):
    """
    Uploads PDF, gets signed URL, runs OCR (inc. images), saves response to JSON.
    """
    logging.info(f"Starting OCR process for: {pdf_path}")
    logging.info(f"Output will be saved to: {output_json_path}")

    if not os.path.isfile(pdf_path):
        logging.error(f"Input PDF file not found at: {pdf_path}")
        sys.exit("Error: Input PDF file does not exist.")

    uploaded_file_id = None
    signed_url_str = None

    try:
        logging.info("Initializing MistralAI client...")
        mistral_client = Mistral(api_key=MISTRAL_API_KEY)
        logging.info("MistralAI client initialized.")

        # --- Step 1: Upload the file ---
        file_basename = os.path.basename(pdf_path)
        logging.info(f"Uploading file '{file_basename}' for purpose '{FILE_UPLOAD_PURPOSE}'...")
        with open(pdf_path, "rb") as pdf_file_handle:
            file_arg_dict = {
                "file_name": file_basename,
                "content": pdf_file_handle,
            }
            file_response = mistral_client.files.upload(
                file=file_arg_dict,
                purpose=FILE_UPLOAD_PURPOSE
            )

            if hasattr(file_response, 'id'):
                uploaded_file_id = file_response.id
                logging.info(f"File uploaded successfully. File ID: {uploaded_file_id}")
            else:
                logging.error(f"Failed to get file ID from upload response. Response: {file_response}")
                sys.exit("Error: File upload response did not contain an ID.")

        # --- Step 2: Get Signed URL ---
        logging.info(f"Getting signed URL for File ID: {uploaded_file_id}")
        signed_url_response = mistral_client.files.get_signed_url(file_id=uploaded_file_id)

        if hasattr(signed_url_response, 'url'):
             signed_url_str = signed_url_response.url
             logging.info(f"Successfully obtained signed URL.")
        else:
             logging.error(f"Failed to get signed URL from response. Response: {signed_url_response}")
             sys.exit("Error: Get signed URL response did not contain a URL.")

        # --- Step 3: Process OCR using the Signed URL ---
        logging.info(f"Sending Signed URL for File ID '{uploaded_file_id}' to MistralAI OCR process (requesting images)...")
        # --- CORRECTED CALL ---
        # Added 'include_image_base64=True' based on documentation example
        ocr_response = mistral_client.ocr.process(
            model=MISTRAL_OCR_MODEL,
            document={
                "type": "document_url",
                "document_url": signed_url_str,
            },
            include_image_base64=True # Ensure image data is included in the response
        )
        # --- END CORRECTION ---
        logging.info("Received OCR response from MistralAI.")

        # --- Process the response ---
        pages_data = []
        if hasattr(ocr_response, 'pages') and ocr_response.pages is not None:
            for page in ocr_response.pages:
                images_data = []
                if hasattr(page, 'images') and page.images:
                    images_data = [
                        {
                            "id": getattr(img, 'id', f'page_{getattr(page, "index", "unknown")}_img_{idx}'),
                            "base64": getattr(img, 'image_base64', None) # Extract base64 if present
                        }
                        for idx, img in enumerate(page.images)
                    ]
                    # Add a check if base64 data is actually present after requesting it
                    if not any(d.get('base64') for d in images_data):
                         logging.warning(f"Requested images for page {getattr(page, 'index', 'unknown')}, but received no base64 data.")
                else:
                     logging.info(f"No 'images' attribute found or images list empty for page {getattr(page, 'index', 'unknown')}.")

                pages_data.append({
                    "index": getattr(page, 'index', -1),
                    "markdown": getattr(page, 'markdown', ""),
                    "images": images_data,
                })
        else:
            logging.warning("OCR response did not contain 'pages' or it was empty.")
            logging.warning(f"Received response structure: {ocr_response}")

        structured_output = {"pages": pages_data}
        logging.info(f"Saving structured OCR data to '{output_json_path}'...")
        with open(output_json_path, 'w', encoding='utf-8') as outfile:
            json.dump(structured_output, outfile, indent=4)
        logging.info(f"Successfully saved OCR data to '{output_json_path}'.")

    # ... (rest of error handling and main block) ...
    except FileNotFoundError as e:
        logging.error(f"File operation error: {e}")
        sys.exit("Error: File operation failed.")
    except AttributeError as ae:
         logging.error(f"An Attribute Error occurred: {ae}", exc_info=True)
         sys.exit("Error: MistralAI client interaction failed (AttributeError).")
    except TypeError as te:
         logging.error(f"A Type Error occurred: {te}", exc_info=True)
         sys.exit("Error: MistralAI client interaction failed (TypeError).")
    except Exception as e:
        if "ValidationError" in str(type(e)):
             logging.error(f"Pydantic Validation Error during API call: {e}", exc_info=True)
             sys.exit("Error: MistralAI client interaction failed (Data Validation).")
        else:
             logging.error(f"An unexpected error occurred: {e}", exc_info=True)
             sys.exit("Error: MistralAI OCR process failed.")

# --- Main execution block ---
if __name__ == "__main__":
    start_time = time.time()
    try:
        run_ocr_on_pdf(PDF_FILE_PATH, OUTPUT_OCR_JSON_FILE)
        logging.info("MistralAI OCR script finished successfully.")
    except SystemExit as se:
        pass
    except Exception as main_e:
        logging.error(f"Script execution failed unexpectedly: {main_e}", exc_info=True)
        sys.exit(1)
    finally:
        end_time = time.time()
        logging.info(f"Total OCR script execution time: {end_time - start_time:.2f} seconds")