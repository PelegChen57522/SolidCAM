import os
import logging
import re
import json
import cohere 
from pinecone import Pinecone, ServerlessSpec, PodSpec 
from dotenv import load_dotenv
import time
import sys
import mimetypes

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API Keys and Environment Variables ---
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "gcp-starter") # Or your Pinecone environment

if not COHERE_API_KEY:
    logging.error("COHERE_API_KEY not found. Please set it in your .env file.")
    sys.exit("Error: Missing COHERE_API_KEY.")
if not PINECONE_API_KEY:
    logging.error("PINECONE_API_KEY not found. Please set it in your .env file.")
    sys.exit("Error: Missing PINECONE_API_KEY.")

# --- Input/Output Files and Pinecone/Cohere Settings ---
OCR_OUTPUT_FILE = "processed_solidcam_doc.json"
PINECONE_INDEX_NAME = "solidcam-chatbot-image-embeddings" # Ensure this matches your Pinecone index
COHERE_MODEL_NAME = 'embed-v4.0' # Multimodal model
PINECONE_DIMENSION = 1024 # Dimension for embed-v4.0
COHERE_INPUT_TYPE_DOC = "search_document" # For document embedding with embed-v4.0
COHERE_EMBEDDING_TYPES = ["float"]

# --- Rate Limiting & Batching ---
API_CALL_DELAY_SECONDS = 0.5
COHERE_BATCH_SIZE = 90 

# --- Control Flags ---
CLEAR_PINECONE_INDEX_BEFORE_RUN = True
CREATE_PINECONE_INDEX_IF_NOT_EXISTS = True
PINECONE_INDEX_TYPE = "serverless" 
PINECONE_CLOUD_PROVIDER = "aws" 
PINECONE_REGION = "us-east-1" 
PINECONE_POD_ENVIRONMENT = "gcp-starter" 
PINECONE_POD_TYPE = "p1.x1" 

# --- Regex Patterns ---
IMAGE_TAG_PATTERN = re.compile(r'!\[(?:.*?)\]\((.*?)\)')
EXCLUDED_HEADERS = ["See Also", "Related Topics"] 
EXCLUDED_HEADERS_PATTERN = "|".join(re.escape(h) for h in EXCLUDED_HEADERS)
HEADER_CHUNK_PATTERN = re.compile(
    rf"^(#{{1,3}}\s+(?!({EXCLUDED_HEADERS_PATTERN})\s*$)(?!-\s)(?!=\s).*?)$",
    re.MULTILINE | re.IGNORECASE
)
INITIAL_TEXT_PATTERN = re.compile(rf"^(.*?)(?={HEADER_CHUNK_PATTERN.pattern}|\Z)", re.DOTALL | re.MULTILINE | re.IGNORECASE)

# --- Helper Functions ---
def create_data_url_from_base64(image_filename_or_id, base64_data_from_ocr):
    if not base64_data_from_ocr: return None
    if base64_data_from_ocr.startswith('data:image'):
        return base64_data_from_ocr 
    mime_type, _ = mimetypes.guess_type(image_filename_or_id)
    if not mime_type:
        mime_type = "image/png" 
        logging.debug(f"Could not determine MIME type for '{image_filename_or_id}'. Defaulting to '{mime_type}'.")
    return f"data:{mime_type};base64,{base64_data_from_ocr}"

def find_images_and_get_data_urls(chunk_text_content, page_level_images_dict):
    found_images_data_urls = {}
    for match in IMAGE_TAG_PATTERN.finditer(chunk_text_content):
        image_filename_in_markdown = match.group(1) 
        if image_filename_in_markdown in page_level_images_dict:
            raw_base64_data = page_level_images_dict[image_filename_in_markdown]
            if raw_base64_data:
                data_url = create_data_url_from_base64(image_filename_in_markdown, raw_base64_data)
                if data_url:
                    found_images_data_urls[image_filename_in_markdown] = data_url
        else:
            logging.warning(f"Image '{image_filename_in_markdown}' referenced in markdown chunk not found in OCR image dictionary for the page. Available keys: {list(page_level_images_dict.keys())}")
    return found_images_data_urls

def chunk_markdown_and_associate_images(page_idx, markdown_content, images_on_page_dict):
    chunks = []
    current_pos = 0
    chunk_on_page_idx = 0
    initial_match = INITIAL_TEXT_PATTERN.match(markdown_content)
    current_header = "" 
    if initial_match:
        text_before_first_header = initial_match.group(1).strip()
        if text_before_first_header:
            chunk_id_str = f"page{page_idx + 1}-chunk{chunk_on_page_idx}"
            image_data_urls_for_chunk = find_images_and_get_data_urls(text_before_first_header, images_on_page_dict)
            chunks.append({
                "id": chunk_id_str, "text": text_before_first_header, "page": page_idx,
                "header": current_header, "images_dataurls": image_data_urls_for_chunk
            })
            chunk_on_page_idx += 1
        current_pos = initial_match.end()

    header_matches_iter = list(HEADER_CHUNK_PATTERN.finditer(markdown_content, pos=current_pos))
    for i, match_obj in enumerate(header_matches_iter):
        current_header = match_obj.group(1).strip()
        content_start_pos = match_obj.end()
        content_end_pos = header_matches_iter[i+1].start() if (i + 1) < len(header_matches_iter) else len(markdown_content)
        text_under_header = markdown_content[content_start_pos:content_end_pos].strip()
        full_chunk_text = f"{current_header}\n\n{text_under_header}".strip()
        if not full_chunk_text: continue
        chunk_id_str = f"page{page_idx + 1}-chunk{chunk_on_page_idx}"
        image_data_urls_for_chunk = find_images_and_get_data_urls(full_chunk_text, images_on_page_dict)
        chunks.append({
            "id": chunk_id_str, "text": full_chunk_text, "page": page_idx,
            "header": current_header, "images_dataurls": image_data_urls_for_chunk
        })
        chunk_on_page_idx += 1
    if not chunks and not markdown_content.strip():
        logging.debug(f"Page {page_idx} resulted in no processable chunks.")
    return chunks

# --- Main Embedding and Storage Function ---
def embed_and_store():
    logging.info(f"Starting multimodal embedding and storage (Cohere Model: {COHERE_MODEL_NAME}, Pinecone Dim: {PINECONE_DIMENSION})...")
    start_time_process = time.time()

    try:
        co = cohere.ClientV2(api_key=COHERE_API_KEY)
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing_indexes_list = pc.list_indexes()
        existing_index_names = [index_info.name for index_info in existing_indexes_list.indexes] if hasattr(existing_indexes_list, 'indexes') else []

        if CREATE_PINECONE_INDEX_IF_NOT_EXISTS and PINECONE_INDEX_NAME not in existing_index_names:
            logging.info(f"Pinecone index '{PINECONE_INDEX_NAME}' not found. Attempting to create...")
            if PINECONE_INDEX_TYPE == "serverless":
                spec = ServerlessSpec(cloud=PINECONE_CLOUD_PROVIDER, region=PINECONE_REGION)
            elif PINECONE_INDEX_TYPE == "pod":
                spec = PodSpec(environment=PINECONE_POD_ENVIRONMENT, pod_type=PINECONE_POD_TYPE, pods=1)
            else:
                logging.error(f"Invalid PINECONE_INDEX_TYPE: {PINECONE_INDEX_TYPE}. Must be 'serverless' or 'pod'.")
                sys.exit(1)
            pc.create_index(name=PINECONE_INDEX_NAME, dimension=PINECONE_DIMENSION, metric="cosine", spec=spec)
            logging.info(f"Index '{PINECONE_INDEX_NAME}' creation initiated. Waiting for initialization...")
            time_to_wait = 120; time_waited = 0
            while time_waited < time_to_wait:
                try:
                    index_description = pc.describe_index(name=PINECONE_INDEX_NAME)
                    if index_description.status == 'Ready':
                        logging.info(f"Index '{PINECONE_INDEX_NAME}' is ready.")
                        break
                    logging.info(f"Index '{PINECONE_INDEX_NAME}' status: {index_description.status}. Waiting...")
                except Exception as desc_e:
                    logging.warning(f"Could not describe index '{PINECONE_INDEX_NAME}' yet: {desc_e}")
                time.sleep(10)
                time_waited += 10
            if time_waited >= time_to_wait:
                logging.error(f"Index '{PINECONE_INDEX_NAME}' did not become ready within {time_to_wait} seconds.")
                sys.exit(1)
        
        index = pc.Index(PINECONE_INDEX_NAME)
        logging.info(f"Cohere (ClientV2) and Pinecone clients initialized. Using index '{PINECONE_INDEX_NAME}'.")
        index_stats = index.describe_index_stats()
        logging.info(f"Current Pinecone index stats: {index_stats}")
        if index_stats.dimension != PINECONE_DIMENSION:
            logging.error(f"CRITICAL: Pinecone index dimension ({index_stats.dimension}) differs from target ({PINECONE_DIMENSION}).")
            sys.exit(1)
    except Exception as e:
        logging.error(f"Client (Cohere/Pinecone) initialization or index handling failed: {e}", exc_info=True)
        sys.exit(1)

    try:
        with open(OCR_OUTPUT_FILE, 'r', encoding='utf-8') as f: ocr_data_from_file = json.load(f)
        logging.info(f"Loaded OCR data from '{OCR_OUTPUT_FILE}'. Contains {len(ocr_data_from_file.get('pages',[]))} pages.")
    except Exception as e:
        logging.error(f"Error loading OCR JSON from '{OCR_OUTPUT_FILE}': {e}", exc_info=True)
        sys.exit(1)

    all_document_chunks = []
    for page_obj in ocr_data_from_file.get("pages", []):
        page_num_idx = page_obj.get("index")
        md_content = page_obj.get("markdown", "")
        images_on_page_dict = {img_meta.get("id"): img_meta.get("base64") 
                               for img_meta in page_obj.get("images", []) 
                               if img_meta.get("id") and img_meta.get("base64")}
        if md_content is None or not isinstance(md_content, str) or page_num_idx is None:
            logging.warning(f"Skipping page (index: {page_num_idx}) due to missing/invalid markdown or index.")
            continue
        chunks_from_page = chunk_markdown_and_associate_images(page_num_idx, md_content, images_on_page_dict)
        all_document_chunks.extend(chunks_from_page)
    logging.info(f"Generated {len(all_document_chunks)} total chunks for embedding.")
    if not all_document_chunks:
        logging.warning("No chunks were generated from OCR data. Exiting.")
        return

    cohere_api_inputs = []; input_index_to_chunk_id_map = {}
    for idx, chunk_data in enumerate(all_document_chunks):
        chunk_id = chunk_data['id']; text_content = chunk_data['text']; image_dataurls_map = chunk_data['images_dataurls'] 
        api_content_list_for_chunk = []
        if text_content and text_content.strip():
            api_content_list_for_chunk.append({"type": "text", "text": text_content.strip()})
        for md_filename, data_url_val in image_dataurls_map.items():
            if data_url_val:
                api_content_list_for_chunk.append({"type": "image_url", "image_url": {"url": data_url_val}})
                logging.debug(f"Adding image '{md_filename}' (as data URL) to content for chunk {chunk_id}")
        if api_content_list_for_chunk: 
            cohere_api_inputs.append({"content": api_content_list_for_chunk})
            input_index_to_chunk_id_map[len(cohere_api_inputs) - 1] = chunk_id
        else:
            logging.warning(f"Chunk {chunk_id} has no text or image content after processing. Skipping.")
    logging.info(f"Prepared {len(cohere_api_inputs)} structured inputs for Cohere embedding.")
    if not cohere_api_inputs:
        logging.warning("No inputs prepared for Cohere. Exiting.")
        return

    successful_embeddings_map = {}; num_embedded = 0; num_failed = 0
    for i in range(0, len(cohere_api_inputs), COHERE_BATCH_SIZE):
        current_batch_api_inputs = cohere_api_inputs[i : i + COHERE_BATCH_SIZE]
        current_batch_chunk_ids = [input_index_to_chunk_id_map[original_idx] for original_idx in range(i, i + len(current_batch_api_inputs))]
        if not current_batch_api_inputs: continue
        try:
            logging.info(f"Embedding batch {i//COHERE_BATCH_SIZE + 1}/{ (len(cohere_api_inputs) -1)//COHERE_BATCH_SIZE + 1 } ({len(current_batch_api_inputs)} items)...")
            api_response = co.embed(model=COHERE_MODEL_NAME, inputs=current_batch_api_inputs, input_type=COHERE_INPUT_TYPE_DOC, embedding_types=COHERE_EMBEDDING_TYPES, output_dimension=PINECONE_DIMENSION)
            response_embeddings = None
            if hasattr(api_response, 'embeddings'):
                if isinstance(api_response.embeddings, list): response_embeddings = api_response.embeddings
                elif hasattr(api_response.embeddings, 'float') and isinstance(api_response.embeddings.float, list): response_embeddings = api_response.embeddings.float
                elif isinstance(api_response.embeddings, dict) and 'float' in api_response.embeddings and isinstance(api_response.embeddings['float'], list): response_embeddings = api_response.embeddings['float']
            if response_embeddings and len(response_embeddings) == len(current_batch_api_inputs):
                for batch_idx, vector in enumerate(response_embeddings):
                    original_chunk_id = current_batch_chunk_ids[batch_idx]
                    successful_embeddings_map[original_chunk_id] = vector
                num_embedded += len(current_batch_api_inputs)
                logging.info(f"Successfully embedded batch of {len(current_batch_api_inputs)} items.")
            else:
                logging.error(f"Embedding response error or mismatch for batch. Expected {len(current_batch_api_inputs)} embeddings. Response: {api_response}")
                num_failed += len(current_batch_api_inputs)
            time.sleep(API_CALL_DELAY_SECONDS)
        except Exception as e:
            logging.error(f"Failed to embed batch: {e}", exc_info=True)
            num_failed += len(current_batch_api_inputs)
    logging.info(f"Total embedding attempts - Success: {num_embedded}, Failures: {num_failed}")

    pinecone_vectors_to_upsert = []
    for chunk_obj in all_document_chunks:
        c_id = chunk_obj['id']
        if c_id in successful_embeddings_map:
            embedding_vec = successful_embeddings_map[c_id]
            img_filenames = list(chunk_obj["images_dataurls"].keys())
            
            meta_payload = {
                "vector_id": c_id, "page": chunk_obj["page"] + 1, "header": chunk_obj.get("header", ""),
                "text_snippet": chunk_obj["text"][:2000], 
                "image_ids": img_filenames, # Store only image IDs (filenames)
                "has_images": len(img_filenames) > 0
            }
            pinecone_vectors_to_upsert.append({"id": c_id, "values": embedding_vec, "metadata": meta_payload})
        else:
            logging.warning(f"No successful embedding found for chunk_id {c_id}. It might have failed or been skipped.")
    logging.info(f"Prepared {len(pinecone_vectors_to_upsert)} vectors for Pinecone upsert (metadata size reduced).")

    if pinecone_vectors_to_upsert:
        logging.info(f"Upserting {len(pinecone_vectors_to_upsert)} vectors to Pinecone index '{PINECONE_INDEX_NAME}'...")
        try:
            pinecone_upsert_batch_size = 20 
            for i in range(0, len(pinecone_vectors_to_upsert), pinecone_upsert_batch_size):
                batch_for_pinecone = pinecone_vectors_to_upsert[i : i + pinecone_upsert_batch_size]
                index.upsert(vectors=batch_for_pinecone) 
                logging.info(f"Upserted batch of {len(batch_for_pinecone)} vectors to Pinecone.")
            logging.info(f"Successfully upserted {len(pinecone_vectors_to_upsert)} vectors.")
            final_stats = index.describe_index_stats()
            logging.info(f"Final Pinecone index stats: {final_stats}")
        except Exception as pinecone_err:
            logging.error(f"Failed during Pinecone upsert: {pinecone_err}", exc_info=True)
    else:
        logging.info("No vectors to upsert to Pinecone.")
    end_time_process = time.time()
    logging.info(f"Embedding and storage process finished in {end_time_process - start_time_process:.2f} seconds.")

# --- Main Execution ---
if __name__ == "__main__":
    script_start_time = time.time()
    logging.info("Starting main script execution for multimodal embedding and storage...")
    if CLEAR_PINECONE_INDEX_BEFORE_RUN:
        logging.warning(f"CLEAR_PINECONE_INDEX_BEFORE_RUN is True. Attempting to clear Pinecone index '{PINECONE_INDEX_NAME}'...")
        try:
            pc_manage = Pinecone(api_key=PINECONE_API_KEY)
            index_list_response = pc_manage.list_indexes()
            current_index_names = [index_info.name for index_info in index_list_response.indexes] if hasattr(index_list_response, 'indexes') else []
            if PINECONE_INDEX_NAME in current_index_names:
                index_to_manage = pc_manage.Index(PINECONE_INDEX_NAME)
                stats_before = index_to_manage.describe_index_stats()
                if stats_before.total_vector_count > 0:
                    logging.info(f"Index '{PINECONE_INDEX_NAME}' has {stats_before.total_vector_count} vectors. Deleting all...")
                    index_to_manage.delete(delete_all=True) 
                    logging.info(f"Clear command (delete_all=True) initiated for index '{PINECONE_INDEX_NAME}'. Waiting ~30s for operation to reflect...")
                    time.sleep(30) 
                else:
                    logging.info(f"Index '{PINECONE_INDEX_NAME}' is already empty.")
                stats_after = index_to_manage.describe_index_stats()
                logging.info(f"Pinecone index stats post-clear attempt: {stats_after}")
            else:
                logging.info(f"Pinecone index '{PINECONE_INDEX_NAME}' not found, skipping clear. It might be created if CREATE_PINECONE_INDEX_IF_NOT_EXISTS is True.")
        except Exception as del_e:
            logging.error(f"Could not clear Pinecone index '{PINECONE_INDEX_NAME}': {del_e}", exc_info=True)
    else:
        logging.info("CLEAR_PINECONE_INDEX_BEFORE_RUN is False. Index will not be cleared.")

    try:
        embed_and_store()
        logging.info("Multimodal embedding and storage script finished successfully.")
    except SystemExit: pass 
    except Exception as main_exec_e:
        logging.error(f"An error occurred during the main execution of embedding script: {main_exec_e}", exc_info=True)
        sys.exit(1) 
    finally:
        script_end_time = time.time()
        logging.info(f"Total script execution time: {script_end_time - script_start_time:.2f} seconds")
