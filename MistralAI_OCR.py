import os
import logging
import json
import time
import mimetypes # <--- Added missing import
from mistralai import Mistral # Ensure mistralai package is installed
from dotenv import load_dotenv
import sys

# --- Configuration ---
load_dotenv() # Load environment variables from .env file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Input/Output Paths ---
PDF_FILE_PATH = "/Users/pelegchen/SolidCAM_ChatBotImageEmbeddings/pdf_files/Milling 2024 Machining Processes.pdf" 

OUTPUT_OCR_JSON_FILE = "processed_solidcam_doc.json" # Output file for structured OCR data

# --- MistralAI Configuration ---
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    logging.error("MISTRAL_API_KEY not found in environment variables. Please set it in your .env file.")
    sys.exit("Error: Missing MISTRAL_API_KEY.")

MISTRAL_OCR_MODEL = "mistral-ocr-latest" 
FILE_UPLOAD_PURPOSE = 'ocr' 

def run_ocr_on_pdf(pdf_path: str, output_json_path: str):
    """
    Uploads a PDF file to Mistral AI, gets a signed URL, runs OCR (including image extraction),
    and saves the structured response (markdown and base64 images per page) to a JSON file.
    """
    logging.info(f"Starting OCR process for PDF: {pdf_path}")
    logging.info(f"Output will be saved to: {output_json_path}")

    # Use os.path.expanduser to handle potential '~' if needed, though the provided path is absolute
    expanded_pdf_path = os.path.expanduser(pdf_path)

    if not os.path.isfile(expanded_pdf_path):
        logging.error(f"Input PDF file not found at the specified path: {expanded_pdf_path}")
        sys.exit(f"Error: Input PDF file does not exist at '{expanded_pdf_path}'. Please check the PDF_FILE_PATH variable.")

    uploaded_file_id = None
    signed_url_str = None

    try:
        logging.info("Initializing MistralAI client...")
        mistral_client = Mistral(api_key=MISTRAL_API_KEY)
        logging.info("MistralAI client initialized.")

        # --- Step 1: Upload the file ---
        file_basename = os.path.basename(expanded_pdf_path)
        logging.info(f"Uploading file '{file_basename}' for purpose '{FILE_UPLOAD_PURPOSE}'...")
        with open(expanded_pdf_path, "rb") as pdf_file_handle:
            # The Mistral Python client expects a dictionary for the 'file' argument
            file_arg_dict = {
                "file_name": file_basename,
                "content": pdf_file_handle, # Pass the file handle directly
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

        # --- Step 2: Get Signed URL for the uploaded file ---
        logging.info(f"Getting signed URL for File ID: {uploaded_file_id}")
        signed_url_response = mistral_client.files.get_signed_url(file_id=uploaded_file_id)

        if hasattr(signed_url_response, 'url'):
             signed_url_str = signed_url_response.url
             logging.info(f"Successfully obtained signed URL.") # Removed URL logging for brevity/security
        else:
             logging.error(f"Failed to get signed URL from response. Response: {signed_url_response}")
             sys.exit("Error: Get signed URL response did not contain a URL.")

        # --- Step 3: Process OCR using the Signed URL ---
        logging.info(f"Sending Signed URL for File ID '{uploaded_file_id}' to MistralAI OCR (requesting images)...")
        ocr_response = mistral_client.ocr.process(
            model=MISTRAL_OCR_MODEL,
            document={
                "type": "document_url",
                "document_url": signed_url_str,
            },
            include_image_base64=True # Crucial for getting image data
        )
        logging.info("Received OCR response from MistralAI.")

        # --- Process the OCR response ---
        pages_data = []
        if hasattr(ocr_response, 'pages') and ocr_response.pages is not None:
            for page_obj in ocr_response.pages: # Renamed 'page' to 'page_obj' to avoid conflict
                images_data_for_page = []
                if hasattr(page_obj, 'images') and page_obj.images:
                    for idx, img_obj in enumerate(page_obj.images):
                        image_id = getattr(img_obj, 'id', f'page_{getattr(page_obj, "index", "unknown")}_img_{idx}')
                        image_base64 = getattr(img_obj, 'image_base64', None)
                        processed_base64 = None # Store the final base64 string (potentially with data URL prefix)
                        if image_base64:
                             # Check if Mistral already provided the data URL prefix
                            if image_base64.startswith('data:image'):
                                processed_base64 = image_base64
                            else:
                                # Attempt to guess mime type from common extensions if ID has one
                                # This is a fallback; Mistral's base64 should ideally be complete.
                                mime_type, _ = mimetypes.guess_type(image_id) # Use the imported module
                                if not mime_type:
                                    mime_type = "image/png" # Default if cannot guess
                                    logging.debug(f"Could not guess mime type for image ID '{image_id}', defaulting to {mime_type}")
                                processed_base64 = f"data:{mime_type};base64,{image_base64}"

                        if processed_base64: # Only add if we have valid base64 data
                            images_data_for_page.append({
                                "id": image_id, # This ID is usually the filename Mistral assigns
                                "base64": processed_base64 # Store the full data URL or processed base64
                            })
                        else:
                            logging.warning(f"Image with ID '{image_id}' on page {getattr(page_obj, 'index', 'unknown')} had empty base64 data.")

                    if images_data_for_page and not any(d.get('base64') for d in images_data_for_page):
                         logging.warning(f"Requested images for page {getattr(page_obj, 'index', 'unknown')}, but received no valid base64 data after processing.")
                else:
                     logging.info(f"No 'images' attribute found or images list empty for page {getattr(page_obj, 'index', 'unknown')}.")

                pages_data.append({
                    "index": getattr(page_obj, 'index', -1), # Page index (0-based)
                    "markdown": getattr(page_obj, 'markdown', ""), # Markdown content of the page
                    "images": images_data_for_page, # List of image dicts for this page
                })
        else:
            logging.warning("OCR response did not contain 'pages' or it was empty.")
            logging.warning(f"Received response structure: {ocr_response}")

        structured_output = {"pages": pages_data} # Final JSON structure
        logging.info(f"Saving structured OCR data to '{output_json_path}'...")
        with open(output_json_path, 'w', encoding='utf-8') as outfile:
            json.dump(structured_output, outfile, indent=4)
        logging.info(f"Successfully saved OCR data to '{output_json_path}'.")

    except FileNotFoundError as e:
        # Log the path that was attempted
        logging.error(f"File operation error (FileNotFound): {e}. Path attempted: {expanded_pdf_path}", exc_info=True)
        sys.exit("Error: File operation failed (FileNotFound). Check paths.")
    except AttributeError as ae:
         logging.error(f"An Attribute Error occurred (likely with Mistral client or response parsing): {ae}", exc_info=True)
         sys.exit("Error: MistralAI client interaction failed (AttributeError).")
    except TypeError as te:
         logging.error(f"A Type Error occurred (likely with Mistral client arguments): {te}", exc_info=True)
         sys.exit("Error: MistralAI client interaction failed (TypeError).")
    except Exception as e: # Catch-all for other unexpected errors
        if "ValidationError" in str(type(e)) or "Pydantic" in str(type(e)): # More specific error for Pydantic
             logging.error(f"Data Validation Error during API call: {e}", exc_info=True)
             sys.exit("Error: MistralAI client interaction failed (Data Validation). Check API compatibility.")
        else:
             logging.error(f"An unexpected error occurred during OCR process: {e}", exc_info=True)
             sys.exit("Error: MistralAI OCR process failed unexpectedly.")

# --- Main execution block ---
if __name__ == "__main__":
    start_time = time.time()
    logging.info("Starting MistralAI OCR script...")
    try:
        run_ocr_on_pdf(PDF_FILE_PATH, OUTPUT_OCR_JSON_FILE)
        logging.info("MistralAI OCR script finished successfully.")
    except SystemExit: 
        pass
    except Exception as main_e:
        logging.error(f"Script execution failed unexpectedly at the main level: {main_e}", exc_info=True)
        sys.exit(1) 
    finally:
        end_time = time.time()
        logging.info(f"Total OCR script execution time: {end_time - start_time:.2f} seconds")
