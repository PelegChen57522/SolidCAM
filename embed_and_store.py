import os
import logging
import re
import json
import cohere
from pinecone import Pinecone
from dotenv import load_dotenv
import uuid
import time
import sys
import mimetypes

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API Keys and Environment Variables ---
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not COHERE_API_KEY: logging.error("COHERE_API_KEY not found"); sys.exit("Error: Missing COHERE_API_KEY.")
if not PINECONE_API_KEY: logging.error("PINECONE_API_KEY not found"); sys.exit("Error: Missing PINECONE_API_KEY.")

# --- Input/Output Files and Pinecone/Cohere Settings ---
OCR_OUTPUT_FILE = "processed_solidcam_doc.json"
PINECONE_INDEX_NAME = "solidcam-chatbot-image-embeddings"
COHERE_MODEL = 'embed-v4.0'
PINECONE_DIMENSION = 1024
COHERE_INPUT_TYPE_DOC = "search_document"
COHERE_EMBEDDING_TYPES = ["float"]

# --- Rate Limiting & Batching ---
API_CALL_DELAY_SECONDS = 0.7
COHERE_BATCH_SIZE = 96

# --- Control Flags ---
# !!! IMPORTANT: Set True to clear old paragraph-based chunks before adding new header-based ones !!!
CLEAR_PINECONE_INDEX_BEFORE_RUN = True

# --- Regex Patterns ---
# Finds Markdown image tags
IMAGE_TAG_PATTERN = re.compile(r'!\[(?:.*?)\]\((.*?)\)')
# --- MODIFIED HEADER PATTERN ---
# Splits text based on H1, H2, H3 headers BUT EXCLUDES specific unwanted headers like "See Also", "Related Topics"
# Uses a negative lookahead `(?!)` and case-insensitive flag `re.IGNORECASE`
EXCLUDED_HEADERS = ["See Also", "Related Topics"] # Add more headers to exclude if needed
EXCLUDED_HEADERS_PATTERN = "|".join(re.escape(h) for h in EXCLUDED_HEADERS) # Creates pattern like: See\\ Also|Related\\ Topics
HEADER_CHUNK_PATTERN = re.compile(
    rf"(^#{{1,3}}\s+(?!({EXCLUDED_HEADERS_PATTERN})\s*$).*$)", # Exclude specific headers
    re.MULTILINE | re.IGNORECASE
)
# Captures text before the first valid header or all text if no valid headers exist
INITIAL_TEXT_PATTERN = re.compile(rf"^(.*?)(?={HEADER_CHUNK_PATTERN.pattern}|\Z)", re.DOTALL | re.MULTILINE | re.IGNORECASE)
# --- END MODIFIED PATTERNS ---


# --- Helper Functions ---
def create_data_url(image_filename, base64_data):
    """Creates a Data URL from base64 data, inferring mime type."""
    if not base64_data: return None
    mime_type, _ = mimetypes.guess_type(image_filename)
    if not mime_type: mime_type = "image/jpeg"; logging.warning(f"Default mime type for '{image_filename}'.")
    if ',' in base64_data: base64_data = base64_data.split(',', 1)[-1]
    return f"data:{mime_type};base64,{base64_data}"

def find_images_in_chunk(chunk_text, images_dict):
    """Finds image tags in text and returns their data {id: data_url}."""
    found_images_data = {}
    for match in IMAGE_TAG_PATTERN.finditer(chunk_text):
        image_filename = match.group(1)
        if image_filename in images_dict:
            base64_str = images_dict[image_filename]
            if base64_str:
                data_url = create_data_url(image_filename, base64_str)
                if data_url: found_images_data[image_filename] = data_url
    return found_images_data

# --- REVERTED Chunking Function (Uses modified HEADER_CHUNK_PATTERN) ---
def chunk_markdown_and_associate_images(page_index, markdown_text, images_dict):
    """
    Chunks markdown text based on VALID headers (excluding specific ones)
    and associates images found within each chunk.
    """
    chunks = []
    current_position = 0
    chunk_id_counter = 0
    page_uuid_base = uuid.uuid4()

    # Handle initial text before the first valid header
    initial_match = INITIAL_TEXT_PATTERN.match(markdown_text)
    if initial_match:
        initial_text = initial_match.group(1).strip()
        if initial_text:
            chunk_id = str(uuid.uuid5(page_uuid_base, f"chunk_{chunk_id_counter}"))
            chunk_images_dataurls = find_images_in_chunk(initial_text, images_dict)
            chunks.append({"text": initial_text, "page": page_index, "id": chunk_id, "images_dataurls": chunk_images_dataurls})
            chunk_id_counter += 1
        current_position = initial_match.end() # Advance position

    # Split by valid headers (H1, H2, H3, excluding specified ones)
    header_matches = list(HEADER_CHUNK_PATTERN.finditer(markdown_text, pos=current_position))

    for i, header_match in enumerate(header_matches):
        header_text = header_match.group(1).strip()
        start_pos = header_match.end()
        end_pos = header_matches[i+1].start() if i + 1 < len(header_matches) else len(markdown_text)
        content_text = markdown_text[start_pos:end_pos].strip()
        # Combine header and its content (including potentially unwanted sub-headers like ## See Also if they fall within this block)
        chunk_text = f"{header_text}\n\n{content_text}".strip()
        if not chunk_text: continue
        chunk_id = str(uuid.uuid5(page_uuid_base, f"chunk_{chunk_id_counter}"))
        chunk_images_dataurls = find_images_in_chunk(chunk_text, images_dict)
        chunks.append({"text": chunk_text, "page": page_index, "id": chunk_id, "images_dataurls": chunk_images_dataurls})
        chunk_id_counter += 1

    # Handle case where no valid headers are found after initial text
    if not header_matches and current_position < len(markdown_text) and not chunks:
        remaining_text = markdown_text[current_position:].strip()
        if remaining_text:
            chunk_id = str(uuid.uuid5(page_uuid_base, f"chunk_{chunk_id_counter}"))
            chunk_images_dataurls = find_images_in_chunk(remaining_text, images_dict)
            chunks.append({"text": remaining_text, "page": page_index, "id": chunk_id, "images_dataurls": chunk_images_dataurls})

    return chunks
# --- END of REVERTED Chunking Function ---


# --- Main Embedding and Storage Function ---
def embed_and_store():
    """Loads data, prepares fused inputs (based on filtered header chunks), embeds, stores."""
    logging.info("Starting embedding and storage process (Filtered Header Chunks)...")
    process_start_time = time.time()

    # --- Initialize Clients ---
    try:
        co = cohere.ClientV2(api_key=COHERE_API_KEY)
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        logging.info("Cohere (ClientV2) and Pinecone clients initialized.")
        index_stats = index.describe_index_stats()
        logging.info(f"Pinecone index '{PINECONE_INDEX_NAME}' stats: {index_stats}")
        if index_stats.get('dimension') != PINECONE_DIMENSION:
             logging.warning(f"Pinecone dimension ({index_stats.get('dimension')}) != Configured Cohere dimension ({PINECONE_DIMENSION}).")
    except Exception as e: logging.error(f"Failed to initialize clients: {e}", exc_info=True); sys.exit(1)

    # --- Load Processed OCR Data ---
    try:
        with open(OCR_OUTPUT_FILE, 'r', encoding='utf-8') as f: ocr_data = json.load(f)
        logging.info(f"Successfully loaded OCR data from '{OCR_OUTPUT_FILE}'.")
    except Exception as e: logging.error(f"Error loading OCR JSON: {e}", exc_info=True); sys.exit(1)

    # --- Process Pages and Generate Chunks (Filtered Header-based) ---
    all_pages_chunks = []
    for page_data in ocr_data.get("pages", []):
        page_index = page_data.get("index"); markdown_text = page_data.get("markdown", "")
        images_list = page_data.get("images", [])
        images_dict = {img.get("id"): img.get("base64") for img in images_list if img.get("id") and img.get("base64")}
        if markdown_text is None or not isinstance(markdown_text, str): continue
        # Use the header-based chunking function (with modified regex)
        page_chunks = chunk_markdown_and_associate_images(page_index, markdown_text, images_dict)
        all_pages_chunks.extend(page_chunks)
    logging.info(f"Generated {len(all_pages_chunks)} total filtered header chunks.") # Chunk count should be <= original header count

    # --- Prepare Inputs, Embed, Prepare Vectors (Logic remains the same) ---
    inputs_to_embed = []; chunk_lookup = {}
    for chunk in all_pages_chunks:
        chunk_id = chunk['id']; chunk_text = chunk['text']; images_dataurls = chunk['images_dataurls']
        content_list = []
        if chunk_text: content_list.append({"type": "text", "text": chunk_text})
        for img_id, data_url in images_dataurls.items():
             if data_url: content_list.append({"type": "image_url", "image_url": {"url": data_url}})
        if content_list:
            inputs_to_embed.append({"id": chunk_id, "input_doc": {"content": content_list}})
            chunk_lookup[chunk_id] = chunk
    logging.info(f"Prepared {len(inputs_to_embed)} inputs for Cohere embedding.")

    embeddings_dict = {}; embedded_count = 0; failed_count = 0
    for i in range(0, len(inputs_to_embed), COHERE_BATCH_SIZE):
        batch_items = inputs_to_embed[i : i + COHERE_BATCH_SIZE]; batch_ids = [item['id'] for item in batch_items]
        batch_inputs = [item['input_doc'] for item in batch_items]
        if not batch_inputs: continue
        try:
            logging.info(f"Embedding batch {i//COHERE_BATCH_SIZE + 1}/{ (len(inputs_to_embed) + COHERE_BATCH_SIZE - 1)//COHERE_BATCH_SIZE }...")
            response = co.embed(model=COHERE_MODEL, inputs=batch_inputs, input_type=COHERE_INPUT_TYPE_DOC,
                                embedding_types=COHERE_EMBEDDING_TYPES, output_dimension=PINECONE_DIMENSION)
            if hasattr(response, 'embeddings') and response.embeddings and hasattr(response.embeddings, 'float') and len(response.embeddings.float) == len(batch_ids):
                embeddings_batch = response.embeddings.float
                for chunk_id, embedding in zip(batch_ids, embeddings_batch): embeddings_dict[chunk_id] = embedding
                embedded_count += len(batch_ids)
            else: logging.error(f"Embedding response error batch {batch_ids[0]}."); failed_count += len(batch_ids)
            time.sleep(API_CALL_DELAY_SECONDS)
        except Exception as embed_e:
            logging.error(f"Failed to embed batch {batch_ids[0]}: {embed_e}", exc_info=False); failed_count += len(batch_ids)
    logging.info(f"Finished embedding attempts. Success: {embedded_count}, Failures: {failed_count}")

    vectors_to_upsert = []
    for chunk_id, embedding in embeddings_dict.items():
        original_chunk_data = chunk_lookup.get(chunk_id)
        if original_chunk_data:
            image_ids = list(original_chunk_data["images_dataurls"].keys())
            metadata = {"page": original_chunk_data["page"], "text_snippet": original_chunk_data["text"][:500],
                        "image_ids": image_ids, "has_images": len(image_ids) > 0}
            vectors_to_upsert.append((chunk_id, embedding, metadata))

    # --- Upsert to Pinecone ---
    if vectors_to_upsert:
        logging.info(f"Upserting {len(vectors_to_upsert)} vectors to Pinecone index '{PINECONE_INDEX_NAME}'...")
        try:
            pinecone_batch_size = 100
            for i in range(0, len(vectors_to_upsert), pinecone_batch_size):
                batch = vectors_to_upsert[i : i + pinecone_batch_size]
                index.upsert(vectors=batch)
            logging.info(f"Successfully upserted {len(vectors_to_upsert)} vectors to Pinecone.")
            final_stats = index.describe_index_stats()
            logging.info(f"Final index stats: {final_stats}")
        except Exception as pinecone_e: logging.error(f"Failed to upsert vectors: {pinecone_e}", exc_info=True)
    else: logging.info("No vectors successfully embedded to upsert.")
    process_end_time = time.time()
    logging.info(f"Embedding and storage function execution time: {process_end_time - process_start_time:.2f} seconds")

# --- Main Execution ---
if __name__ == "__main__":
    main_start_time = time.time()
    logging.info("Starting main script execution (Filtered Header Chunking)...")
    if CLEAR_PINECONE_INDEX_BEFORE_RUN:
         logging.warning(f"CLEAR_PINECONE_INDEX_BEFORE_RUN is True. Clearing index '{PINECONE_INDEX_NAME}'...")
         try:
             pc_mgmt = Pinecone(api_key=PINECONE_API_KEY)
             if PINECONE_INDEX_NAME in [idx['name'] for idx in pc_mgmt.list_indexes().get('indexes',[])]:
                 index_to_clear = pc_mgmt.Index(PINECONE_INDEX_NAME)
                 logging.info("Clearing all vectors..."); index_to_clear.delete(delete_all=True)
                 logging.info(f"Index clear initiated. Waiting..."); time.sleep(15)
                 stats_after_clear = index_to_clear.describe_index_stats(); logging.info(f"Index stats after clear: {stats_after_clear}")
             else: logging.info(f"Index '{PINECONE_INDEX_NAME}' not found, skipping clear.")
         except Exception as delete_e: logging.error(f"Could not clear Pinecone index: {delete_e}", exc_info=True)
    else:
        logging.info("CLEAR_PINECONE_INDEX_BEFORE_RUN is False. Index will not be cleared.")

    try:
        embed_and_store()
        logging.info("Embedding and storage script finished successfully.")
    except SystemExit: pass
    except Exception as e:
        logging.error(f"Embedding script failed: {e}", exc_info=True); sys.exit(1)
    finally:
        main_end_time = time.time()
        logging.info(f"Total script execution time: {main_end_time - main_start_time:.2f} seconds")