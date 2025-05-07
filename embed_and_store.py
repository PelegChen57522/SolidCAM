
import os
import logging
import re
import json
import cohere
from pinecone import Pinecone
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

if not COHERE_API_KEY:
    logging.error("COHERE_API_KEY not found in environment variables. Please set it in your .env file.")
    sys.exit("Error: Missing COHERE_API_KEY.")
if not PINECONE_API_KEY:
    logging.error("PINECONE_API_KEY not found in environment variables. Please set it in your .env file.")
    sys.exit("Error: Missing PINECONE_API_KEY.")

# --- Input/Output Files and Pinecone/Cohere Settings ---
OCR_OUTPUT_FILE = "processed_solidcam_doc.json" # Make sure this file exists and is populated
PINECONE_INDEX_NAME = "solidcam-chatbot-image-embeddings"
COHERE_MODEL_NAME = 'embed-v4.0'
PINECONE_DIMENSION = 1024 # Ensure this matches your Pinecone index and query embedding dimension
COHERE_INPUT_TYPE_DOC = "search_document" # For document embedding
COHERE_EMBEDDING_TYPES = ["float"]

# --- Rate Limiting & Batching ---
API_CALL_DELAY_SECONDS = 0.7 # Be mindful of API rate limits
COHERE_BATCH_SIZE = 90 # Cohere API batch limit (e.g., 96 for embed-v4.0 with ClientV2)

# --- Control Flags ---
# !!! SET TO TRUE to re-populate Pinecone with the new metadata structure !!!
CLEAR_PINECONE_INDEX_BEFORE_RUN = True

# --- Regex Patterns ---
IMAGE_TAG_PATTERN = re.compile(r'!\[(?:.*?)\]\((.*?)\)')
EXCLUDED_HEADERS = ["See Also", "Related Topics"] # Headers to ignore for chunking
EXCLUDED_HEADERS_PATTERN = "|".join(re.escape(h) for h in EXCLUDED_HEADERS)
HEADER_CHUNK_PATTERN = re.compile(
    rf"^(#{{1,3}}\s+(?!({EXCLUDED_HEADERS_PATTERN})\s*$)(?!-\s).*?)$", # Valid headers (H1, H2, H3)
    re.MULTILINE | re.IGNORECASE
)
INITIAL_TEXT_PATTERN = re.compile(rf"^(.*?)(?={HEADER_CHUNK_PATTERN.pattern}|\Z)", re.DOTALL | re.MULTILINE | re.IGNORECASE)

# --- Helper Functions ---
def create_data_url(image_filename, base64_data):
    """Creates a Data URL from a base64 string and filename."""
    if not base64_data: return None
    mime_type, _ = mimetypes.guess_type(image_filename)
    if not mime_type:
        mime_type = "image/jpeg" # Default if type can't be guessed
        logging.warning(f"Could not determine MIME type for '{image_filename}'. Defaulting to '{mime_type}'.")
    if ',' in base64_data: # Strip prefix if present
        base64_data = base64_data.split(',', 1)[-1]
    return f"data:{mime_type};base64,{base64_data}"

def find_images_in_chunk(chunk_text, images_dict):
    """Finds all image tags in a markdown chunk and returns their data URLs."""
    found_images_data = {}
    for match in IMAGE_TAG_PATTERN.finditer(chunk_text):
        image_filename = match.group(1)
        if image_filename in images_dict:
            base64_str = images_dict[image_filename]
            if base64_str:
                data_url = create_data_url(image_filename, base64_str)
                if data_url: found_images_data[image_filename] = data_url
    return found_images_data

def chunk_markdown_and_associate_images(page_index, markdown_text, images_dict):
    """
    Chunks markdown based on H1, H2, H3 headers, excluding specified non-content headers.
    Associates images found within each chunk.
    Generates human-readable IDs like page<P>-chunk<C>.
    Returns list of chunk objects including the header text.
    """
    chunks = []
    current_position = 0
    chunk_index_on_page = 0
    initial_match = INITIAL_TEXT_PATTERN.match(markdown_text)
    header_text = "" # Default header for initial text if no specific header found
    if initial_match:
        initial_text = initial_match.group(1).strip()
        if initial_text:
            chunk_id = f"page{page_index + 1}-chunk{chunk_index_on_page}"
            chunk_images_dataurls = find_images_in_chunk(initial_text, images_dict)
            chunks.append({"text": initial_text, "page": page_index, "id": chunk_id,
                           "images_dataurls": chunk_images_dataurls, "header": header_text})
            chunk_index_on_page += 1
        current_position = initial_match.end()

    header_matches = list(HEADER_CHUNK_PATTERN.finditer(markdown_text, pos=current_position))
    for i, header_match in enumerate(header_matches):
        header_text = header_match.group(1).strip() # This is the actual header like "# Introduction"
        start_pos = header_match.end()
        end_pos = header_matches[i+1].start() if i + 1 < len(header_matches) else len(markdown_text)
        content_text = markdown_text[start_pos:end_pos].strip()
        chunk_text = f"{header_text}\n\n{content_text}".strip() # Include header in chunk text
        if not chunk_text: continue # Skip if chunk ends up empty
        chunk_id = f"page{page_index + 1}-chunk{chunk_index_on_page}"
        chunk_images_dataurls = find_images_in_chunk(chunk_text, images_dict)
        chunks.append({"text": chunk_text, "page": page_index, "id": chunk_id,
                       "images_dataurls": chunk_images_dataurls, "header": header_text})
        chunk_index_on_page += 1

    if not header_matches and current_position < len(markdown_text) and not chunks: # Trailing content
        remaining_text = markdown_text[current_position:].strip()
        if remaining_text:
            chunk_id = f"page{page_index + 1}-chunk{chunk_index_on_page}"
            chunk_images_dataurls = find_images_in_chunk(remaining_text, images_dict)
            chunks.append({"text": remaining_text, "page": page_index, "id": chunk_id,
                           "images_dataurls": chunk_images_dataurls, "header": ""})
    return chunks

# --- Main Embedding and Storage Function ---
def embed_and_store():
    """Loads data, prepares inputs, embeds using Cohere ClientV2, and stores in Pinecone."""
    logging.info(f"Starting embedding and storage process (Target Dimension: {PINECONE_DIMENSION}) using Cohere ClientV2 structure...")
    process_start_time = time.time()

    # Initialize Cohere and Pinecone clients
    try:
        co = cohere.ClientV2(api_key=COHERE_API_KEY) # Using ClientV2 as it worked for output_dimension
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        logging.info("Cohere (ClientV2) and Pinecone clients initialized.")
        # Check current index dimension if it exists
        try:
            index_stats = index.describe_index_stats()
            logging.info(f"Current Pinecone index stats: {index_stats}")
            if index_stats.dimension != PINECONE_DIMENSION and CLEAR_PINECONE_INDEX_BEFORE_RUN:
                logging.warning(f"Pinecone index dimension ({index_stats.dimension}) differs from target ({PINECONE_DIMENSION}) and will be cleared.")
            elif index_stats.dimension != PINECONE_DIMENSION: # If not clearing and dimensions mismatch
                 logging.error(f"CRITICAL: Pinecone index dimension ({index_stats.dimension}) differs from target ({PINECONE_DIMENSION}), and CLEAR_PINECONE_INDEX_BEFORE_RUN is False. This will lead to errors. Please clear the index or match dimensions.")
                 sys.exit(1) # Exit if dimensions mismatch and not clearing
        except Exception as e: # Catch errors if index doesn't exist yet or other issues
            logging.warning(f"Could not fetch Pinecone index stats: {e}. This is okay if index doesn't exist yet and will be created by upsert (for serverless) or needs manual creation for pod-based.")
    except Exception as e:
        logging.error(f"Client initialization failed: {e}", exc_info=True)
        sys.exit(1)

    # Load OCR data
    try:
        with open(OCR_OUTPUT_FILE, 'r', encoding='utf-8') as f: ocr_data = json.load(f)
        logging.info(f"Loaded OCR data from '{OCR_OUTPUT_FILE}'.")
    except FileNotFoundError:
        logging.error(f"OCR output file '{OCR_OUTPUT_FILE}' not found. Please run the OCR script first.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading OCR JSON: {e}", exc_info=True)
        sys.exit(1)

    # Generate chunks from OCR data
    all_pages_chunks = []
    for page_data in ocr_data.get("pages", []):
        page_index = page_data.get("index")
        markdown_text = page_data.get("markdown", "")
        images_list = page_data.get("images", [])
        images_dict = {img.get("id"): img.get("base64") for img in images_list if img.get("id") and img.get("base64")}
        if markdown_text is None or not isinstance(markdown_text, str):
            logging.warning(f"Skipping page {page_index} due to missing or invalid markdown text.")
            continue
        if page_index is None: # Page index is crucial for ID generation
            logging.warning(f"Skipping page with missing index. Markdown: {markdown_text[:100]}...")
            continue
        page_chunks = chunk_markdown_and_associate_images(page_index, markdown_text, images_dict)
        all_pages_chunks.extend(page_chunks)
    logging.info(f"Generated {len(all_pages_chunks)} total chunks.")
    if not all_pages_chunks:
        logging.warning("No chunks were generated from the OCR data. Exiting.")
        return

    # Prepare inputs for Cohere ClientV2 embed method
    inputs_for_cohere_v2 = []
    chunk_lookup = {} # To map chunk IDs back to original chunk data after embedding
    for chunk in all_pages_chunks:
        chunk_id = chunk['id']
        chunk_text = chunk['text']
        content_list = []
        if chunk_text: # Ensure there is text to embed
            content_list.append({"type": "text", "text": chunk_text})
        # If you plan to add multimodal inputs (e.g., image URLs) for embed-v4.0,
        # they would be added to content_list here, e.g.:
        # for img_filename, data_url in chunk['images_dataurls'].items():
        #     if data_url: content_list.append({"type": "image_url", "image_url": {"url": data_url}})
        if content_list: # Only add if there's content
            inputs_for_cohere_v2.append({"id": chunk_id, "input_doc_structure": {"content": content_list}})
            chunk_lookup[chunk_id] = chunk
    logging.info(f"Prepared {len(inputs_for_cohere_v2)} inputs for Cohere ClientV2 embedding.")

    # Embed chunks using Cohere ClientV2
    embeddings_dict = {} # {chunk_id: embedding_vector}
    embedded_count = 0
    failed_count = 0
    for i in range(0, len(inputs_for_cohere_v2), COHERE_BATCH_SIZE):
        batch_items_v2 = inputs_for_cohere_v2[i : i + COHERE_BATCH_SIZE]
        batch_ids_v2 = [item['id'] for item in batch_items_v2]
        batch_inputs_for_api = [item['input_doc_structure'] for item in batch_items_v2] # Get the structured input part
        if not batch_inputs_for_api: continue # Skip if batch is empty
        try:
            logging.info(f"Embedding batch (ClientV2) starting with ID {batch_ids_v2[0]} ({len(batch_inputs_for_api)} items)...")
            response = co.embed(
                model=COHERE_MODEL_NAME,
                inputs=batch_inputs_for_api, # Pass the list of structured inputs
                input_type=COHERE_INPUT_TYPE_DOC,
                embedding_types=COHERE_EMBEDDING_TYPES,
                output_dimension=PINECONE_DIMENSION # This worked with ClientV2
            )
            # Access embeddings from response.embeddings.float for ClientV2
            if hasattr(response, 'embeddings') and response.embeddings and \
               hasattr(response.embeddings, 'float') and \
               len(response.embeddings.float) == len(batch_ids_v2):
                embeddings_batch = response.embeddings.float
                for chunk_id, embedding in zip(batch_ids_v2, embeddings_batch):
                    embeddings_dict[chunk_id] = embedding
                embedded_count += len(batch_ids_v2)
            else:
                logging.error(f"Embedding response error or mismatch for batch (ClientV2) {batch_ids_v2[0]}. Response: {response}")
                failed_count += len(batch_ids_v2)
            time.sleep(API_CALL_DELAY_SECONDS) # Respect API rate limits
        except Exception as embed_e:
            logging.error(f"Failed to embed batch (ClientV2) {batch_ids_v2[0]}: {embed_e}", exc_info=False) # exc_info=False for cleaner logs on repeated errors
            failed_count += len(batch_ids_v2)
    logging.info(f"Embedding attempts (ClientV2). Success: {embedded_count}, Failures: {failed_count}")

    # Prepare vectors for Pinecone upsert
    vectors_to_upsert = []
    for chunk_id, embedding in embeddings_dict.items():
        original_chunk_data = chunk_lookup.get(chunk_id)
        if original_chunk_data:
            image_filenames = [os.path.basename(fn) for fn in original_chunk_data["images_dataurls"].keys()]
            metadata = {
                "vector_id": chunk_id, # Store the Pinecone vector ID (which is our chunk_id) in metadata
                "page": original_chunk_data["page"] + 1, # 1-based page number
                "header": original_chunk_data.get("header", ""),
                "text_snippet": original_chunk_data["text"][:1000], # For PineconeVectorStore text_key
                "image_ids": image_filenames,
                "has_images": len(image_filenames) > 0
            }
            vectors_to_upsert.append((chunk_id, embedding, metadata)) # Pinecone vector ID, embedding, metadata
    logging.info(f"Prepared {len(vectors_to_upsert)} vectors for Pinecone.")

    # Upsert vectors to Pinecone
    if vectors_to_upsert:
        logging.info(f"Upserting {len(vectors_to_upsert)} vectors to Pinecone index '{PINECONE_INDEX_NAME}'...")
        try:
            pinecone_batch_size = 100 # Pinecone recommends batches of 100 or fewer for upsert
            for i in range(0, len(vectors_to_upsert), pinecone_batch_size):
                batch_to_upsert = vectors_to_upsert[i : i + pinecone_batch_size]
                index.upsert(vectors=batch_to_upsert)
                logging.info(f"Upserted batch of {len(batch_to_upsert)} vectors to Pinecone.")
            logging.info(f"Successfully upserted {len(vectors_to_upsert)} vectors.")
            final_stats_after_upsert = index.describe_index_stats()
            logging.info(f"Final Pinecone index stats after upsert: {final_stats_after_upsert}")
        except Exception as pinecone_e:
            logging.error(f"Failed during Pinecone upsert: {pinecone_e}", exc_info=True)
    else:
        logging.info("No vectors to upsert.")
    process_end_time = time.time()
    logging.info(f"Embedding and storage function finished in {process_end_time - process_start_time:.2f} seconds.")

# --- Main Execution ---
if __name__ == "__main__":
    main_start_time = time.time()
    logging.info("Starting main script execution (using ClientV2 structure for embedding)...")

    if CLEAR_PINECONE_INDEX_BEFORE_RUN:
         logging.warning(f"Attempting to clear Pinecone index '{PINECONE_INDEX_NAME}' as CLEAR_PINECONE_INDEX_BEFORE_RUN is True...")
         try:
             pc_mgmt = Pinecone(api_key=PINECONE_API_KEY)
             existing_indexes = [idx_spec.name for idx_spec in pc_mgmt.list_indexes()] # Get list of index names
             if PINECONE_INDEX_NAME in existing_indexes:
                 index_to_clear = pc_mgmt.Index(PINECONE_INDEX_NAME)
                 # Check if index is empty before trying to delete all (to avoid 404 on empty default namespace)
                 stats_before_clear = index_to_clear.describe_index_stats()
                 if stats_before_clear.total_vector_count > 0:
                     index_to_clear.delete(delete_all=True) # Clears all vectors in the default namespace
                     logging.info(f"Clear command (delete_all=True) initiated for index '{PINECONE_INDEX_NAME}'. Waiting ~30s for operation to reflect...")
                     time.sleep(30) # Give some time for the delete_all operation
                 else:
                     logging.info(f"Index '{PINECONE_INDEX_NAME}' is already empty. No need to call delete_all.")
                 stats_after_clear = index_to_clear.describe_index_stats() # Check stats again
                 logging.info(f"Pinecone index stats post-clear attempt: {stats_after_clear}")
             else:
                 logging.info(f"Pinecone index '{PINECONE_INDEX_NAME}' not found, skipping clear. It might be created on first upsert (serverless) or needs manual creation (pod-based).")
         except Exception as delete_e:
             logging.error(f"Could not clear Pinecone index '{PINECONE_INDEX_NAME}': {delete_e}", exc_info=True)
    else:
        logging.info("CLEAR_PINECONE_INDEX_BEFORE_RUN is False. Index will not be cleared.")

    try:
        embed_and_store()
        logging.info("Embedding and storage script finished successfully.")
    except SystemExit: # Allow sys.exit() to terminate script gracefully
        pass # Already logged by sys.exit call
    except Exception as e:
        logging.error(f"An error occurred during the main execution of embedding script: {e}", exc_info=True)
        sys.exit(1) # Exit with error code
    finally:
        main_end_time = time.time()
        logging.info(f"Total script execution time: {main_end_time - main_start_time:.2f} seconds")
