import os
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any
import sys
import json # To load OCR data

import cohere # Import native SDK
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS


from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory
from langchain_cohere import CohereRerank, ChatCohere
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, message_to_dict

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API Keys ---
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not COHERE_API_KEY: logging.critical("COHERE_API_KEY not found.")
if not PINECONE_API_KEY: logging.critical("PINECONE_API_KEY not found.")

# --- File Paths ---
OCR_OUTPUT_FILE = "processed_solidcam_doc.json"

# --- Pinecone Settings ---
PINECONE_INDEX_NAME = "solidcam-chatbot-image-embeddings"

# --- Cohere Settings ---
COHERE_EMBED_MODEL_NAME = 'embed-v4.0' # For retrieval embeddings
COHERE_TARGET_EMBED_DIMENSION = 1024
COHERE_GEN_MODEL = 'c4ai-aya-vision-32b' # Larger Vision Model
COHERE_RERANK_MODEL = 'rerank-english-v3.0'
COHERE_INPUT_TYPE_DOC_FOR_V2_EMBED = "search_document"
COHERE_INPUT_TYPE_QUERY_FOR_V2_EMBED = "search_query"
COHERE_EMBEDDING_TYPES_FOR_V2_EMBED = ["float"]

# --- RAG Settings ---
RERANK_TOP_N = 3 # Number of documents after reranking
INITIAL_RETRIEVAL_K = 10 # Initial number of documents to fetch
MAX_IMAGES_PER_PROMPT = 4 # Cohere API limit for Aya Vision Chat

# --- Load OCR Data for Image Retrieval ---
ocr_data_store = {}
try:
    with open(OCR_OUTPUT_FILE, 'r', encoding='utf-8') as f:
        loaded_ocr_data = json.load(f)
        for page_data in loaded_ocr_data.get("pages", []):
            page_idx = page_data.get("index") # 0-based index from OCR
            if page_idx is not None:
                ocr_data_store[page_idx] = {
                    img.get("id"): img.get("base64")
                    for img in page_data.get("images", []) if img.get("id") and img.get("base64")
                }
    logging.info(f"Successfully loaded OCR data for image retrieval from {OCR_OUTPUT_FILE}")
except Exception as e:
    logging.error(f"Failed to load OCR data from {OCR_OUTPUT_FILE}. Image retrieval will not work. Error: {e}", exc_info=True)
    ocr_data_store = {}

# --- Custom Langchain-compatible Cohere Embeddings Class (Unchanged) ---
class CustomCohereEmbeddingsWithClientV2(Embeddings):
    # ... (Keep the class definition exactly as before) ...
    client: cohere.ClientV2
    model: str
    output_dimension: int
    embedding_types: List[str]
    input_type: str

    def __init__(self, api_key: str, model_name: str, output_dim: int, embedding_types: List[str], input_type_for_embedding: str):
        super().__init__()
        self.client = cohere.ClientV2(api_key=api_key)
        self.model = model_name
        self.output_dimension = output_dim
        self.embedding_types = embedding_types
        self.input_type = input_type_for_embedding

    def _prepare_inputs_for_clientv2(self, texts: List[str]) -> List[Dict[str, Any]]:
        structured_inputs = []
        for text_item in texts:
            if text_item and text_item.strip():
                content_list = [{"type": "text", "text": text_item.strip()}]
                structured_inputs.append({"content": content_list})
            else: logging.debug(f"Skipping empty text item: '{text_item}'")
        return structured_inputs

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts: return [[] for _ in texts]
        original_input_type = self.input_type
        self.input_type = COHERE_INPUT_TYPE_DOC_FOR_V2_EMBED
        prepared_inputs = self._prepare_inputs_for_clientv2(texts)
        self.input_type = original_input_type
        if not prepared_inputs: return [[] for _ in texts]
        try:
            response = self.client.embed(model=self.model, inputs=prepared_inputs, input_type=COHERE_INPUT_TYPE_DOC_FOR_V2_EMBED, embedding_types=self.embedding_types, output_dimension=self.output_dimension)
            embeddings = None
            if hasattr(response, 'embeddings'):
                if isinstance(response.embeddings, list): embeddings = response.embeddings
                elif hasattr(response.embeddings, 'float') and isinstance(response.embeddings.float, list): embeddings = response.embeddings.float
                elif isinstance(response.embeddings, dict) and 'float' in response.embeddings: embeddings = response.embeddings['float']
            if embeddings and len(embeddings) == len(prepared_inputs):
                valid_texts_map = {text: emb for text, emb in zip([t for t in texts if t and t.strip()], embeddings)}
                return [valid_texts_map.get(text, []) for text in texts]
            return [[] for _ in texts]
        except Exception as e:
            logging.error(f"Error embedding documents: {e}", exc_info=True)
            return [[] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        if not text or not text.strip(): return []
        prepared_inputs = self._prepare_inputs_for_clientv2([text])
        if not prepared_inputs: return []
        try:
            response = self.client.embed(model=self.model, inputs=prepared_inputs, input_type=COHERE_INPUT_TYPE_QUERY_FOR_V2_EMBED, embedding_types=self.embedding_types, output_dimension=self.output_dimension)
            embeddings = None
            if hasattr(response, 'embeddings'):
                if isinstance(response.embeddings, list) and response.embeddings: embeddings = response.embeddings[0]
                elif hasattr(response.embeddings, 'float') and isinstance(response.embeddings.float, list) and response.embeddings.float: embeddings = response.embeddings.float[0]
                elif isinstance(response.embeddings, dict) and 'float' in response.embeddings and response.embeddings['float']: embeddings = response.embeddings['float'][0]
            return embeddings if embeddings else []
        except Exception as e:
            logging.error(f"Error embedding query: {e}", exc_info=True)
            return []


# --- Global Chatbot Components Initialization ---
vectorstore = None
compression_retriever = None
llm = None # Langchain LLM wrapper
memory = None
cohere_native_client = None # Native Cohere client for fallback

def initialize_chatbot_components():
    """Initializes individual Langchain components and native Cohere client."""
    global vectorstore, compression_retriever, llm, memory, cohere_native_client
    if not COHERE_API_KEY or not PINECONE_API_KEY:
        logging.error("API keys missing. Cannot initialize components.")
        return False

    try:
        cohere_native_client = cohere.ClientV2(api_key=COHERE_API_KEY)
        logging.info("Initialized native Cohere ClientV2.")
        custom_query_embeddings = CustomCohereEmbeddingsWithClientV2(
            api_key=COHERE_API_KEY, model_name=COHERE_EMBED_MODEL_NAME,
            output_dim=COHERE_TARGET_EMBED_DIMENSION, embedding_types=COHERE_EMBEDDING_TYPES_FOR_V2_EMBED,
            input_type_for_embedding=COHERE_INPUT_TYPE_QUERY_FOR_V2_EMBED
        )
        logging.info(f"Connecting to Pinecone index: {PINECONE_INDEX_NAME}")
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME, embedding=custom_query_embeddings, text_key="text_snippet"
        )
        logging.info("Connected to Pinecone.")
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": INITIAL_RETRIEVAL_K})
        logging.info(f"Base retriever initialized (k={INITIAL_RETRIEVAL_K}).")
        logging.info(f"Initializing Cohere Reranker: {COHERE_RERANK_MODEL}")
        reranker = CohereRerank(model=COHERE_RERANK_MODEL, top_n=RERANK_TOP_N, cohere_api_key=COHERE_API_KEY)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=base_retriever
        )
        logging.info(f"Compression Retriever initialized (top_n={RERANK_TOP_N}).")
        logging.info(f"Initializing Langchain ChatCohere wrapper for Vision model: {COHERE_GEN_MODEL}")
        llm = ChatCohere(model=COHERE_GEN_MODEL, temperature=0.2, cohere_api_key=COHERE_API_KEY)
        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True, output_key='answer'
        )
        logging.info("Conversation memory initialized.")
        logging.info("Chatbot components initialized successfully.")
        return True
    except Exception as e:
        logging.error(f"Fatal error during Langchain/Cohere component initialization: {e}", exc_info=True)
        vectorstore = compression_retriever = llm = memory = cohere_native_client = None
        return False

# --- Flask App Setup ---
app = Flask(__name__, template_folder='templates')
CORS(app)
components_initialized = initialize_chatbot_components()

def get_image_data_url(page_idx: int, image_id: str) -> str | None:
    """Retrieves the base64 data URL for an image from the loaded OCR store."""
    if not ocr_data_store:
        logging.warning("OCR data store not loaded, cannot retrieve image data.")
        return None
    page_images = ocr_data_store.get(page_idx - 1)
    if page_images:
        data_url = page_images.get(image_id)
        if data_url: return data_url
        else: logging.warning(f"Image ID '{image_id}' found on page {page_idx}, but base64 data is missing in OCR store.")
    else: logging.warning(f"Page index {page_idx - 1} not found in OCR store for image ID '{image_id}'.")
    return None

def format_context_for_aya_vision(retrieved_docs: List[Any]) -> List[Dict[str, Any]]:
    """
    Formats retrieved documents into a list of content parts (text and image URLs)
    suitable for Aya Vision's Chat input, enforcing the image limit AND
    PRIORITIZING IMAGES ONLY FROM THE TOP-RANKED DOCUMENT.
    Adds clearer labels to text parts.
    """
    formatted_context_parts = []
    added_image_urls = set()
    images_added_count = 0
    MAX_IMAGES = MAX_IMAGES_PER_PROMPT

    logging.info(f"Formatting context from {len(retrieved_docs)} retrieved documents (Max Images: {MAX_IMAGES}, Prioritize Rank 1).")
    for rank, doc in enumerate(retrieved_docs): # Iterate in reranked order
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        text_snippet = metadata.get('text_snippet', doc.page_content if hasattr(doc, 'page_content') else "")
        page_num = metadata.get('page')
        image_ids = metadata.get('image_ids', [])
        has_images_flag = metadata.get('has_images', False)
        image_added_for_this_doc = False # Flag to track if image was added for this specific doc

        # --- Add Image Parts (ONLY if rank is 0 and limit not reached) ---
        if rank == 0 and page_num is not None and image_ids and has_images_flag:
            logging.debug(f"Processing images for Top-Ranked Doc (Page {page_num}): {image_ids}")
            for img_id in image_ids:
                if images_added_count >= MAX_IMAGES:
                    logging.warning(f"Reached max image limit ({MAX_IMAGES}) while processing top document. Skipping further images including '{img_id}'.")
                    break # Stop checking images for this top document

                data_url = get_image_data_url(page_num, img_id)
                if data_url and data_url not in added_image_urls:
                    # Add image part FIRST for this document
                    formatted_context_parts.append({"type": "image_url", "image_url": {"url": data_url}})
                    added_image_urls.add(data_url)
                    images_added_count += 1
                    image_added_for_this_doc = True
                    logging.info(f"Added image {img_id} from page {page_num} (Rank 1) to LLM context (Image {images_added_count}/{MAX_IMAGES}).")
                elif data_url is None: logging.warning(f"Could not retrieve data URL for image ID '{img_id}' on page {page_num}.")
                elif data_url in added_image_urls: logging.debug(f"Skipping duplicate image data URL for image ID '{img_id}'")
        elif rank > 0 and has_images_flag:
             logging.debug(f"Skipping images from lower-ranked document (Rank {rank+1}, Page {page_num}) due to prioritization.")

        # --- Add Text Part (with clearer labeling) ---
        if text_snippet and text_snippet.strip():
            source_header = metadata.get('header', '').strip().replace("#", "").strip()
            source_prefix = f"[Source Document {rank+1}: Page {page_num or 'N/A'}"
            if source_header: source_prefix += f", Section: {source_header[:50]}{'...' if len(source_header) > 50 else ''}"
            source_prefix += "]"
            
            # Modify the text content to explicitly label text and associated images
            labeled_text_content = f"{source_prefix}\n--- Text Context ---\n{text_snippet.strip()}"
            if image_added_for_this_doc: # Only mention associated images if they were actually added
                 labeled_text_content += f"\n--- Associated Image(s) Provided Above ---"
            elif rank == 0 and has_images_flag and not image_added_for_this_doc: # Top doc had images but limit reached/failed
                 labeled_text_content += f"\n--- Note: Associated image(s) existed but were not included due to limits/errors ---"

            formatted_context_parts.append({"type": "text", "text": labeled_text_content})
            logging.debug(f"Added labeled text snippet from page {page_num} (Rank {rank+1})")


    # --- Limit Total Context Parts ---
    MAX_TOTAL_CONTEXT_PARTS = 15
    if len(formatted_context_parts) > MAX_TOTAL_CONTEXT_PARTS:
        logging.warning(f"Total context parts {len(formatted_context_parts)} exceeds limit {MAX_TOTAL_CONTEXT_PARTS}. Truncating.")
        formatted_context_parts = formatted_context_parts[:MAX_TOTAL_CONTEXT_PARTS]
    logging.info(f"Formatted context contains {len(formatted_context_parts)} parts ({images_added_count} images) for LLM.")
    return formatted_context_parts

def convert_langchain_messages_to_cohere_format(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """Converts Langchain message objects to the dictionary format expected by Cohere's native chat API."""
    # ... (Keep this function exactly as before) ...
    cohere_messages = []
    for msg in messages:
        role = ""; content = None
        if isinstance(msg, HumanMessage):
            role = "user"
            if isinstance(msg.content, list): content = msg.content
            elif isinstance(msg.content, str): content = [{"type": "text", "text": msg.content}]
            else: content = [{"type": "text", "text": str(msg.content)}]
        elif isinstance(msg, AIMessage):
            role = "assistant"
            if isinstance(msg.content, str): content = [{"type": "text", "text": msg.content}]
            else: content = [{"type": "text", "text": str(msg.content)}]
        elif isinstance(msg, SystemMessage):
            role = "system"
            content = [{"type": "text", "text": msg.content}]
        if role and content: cohere_messages.append({"role": role, "content": content})
        else: logging.warning(f"Could not convert Langchain message to Cohere format: {msg}")
    return cohere_messages


@app.route('/')
def index_route():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat_route():
    global components_initialized, compression_retriever, llm, memory, cohere_native_client
    
    if not components_initialized:
        logging.error("Chatbot components not initialized.")
        return jsonify({"error": "Chatbot is not available. Initialization failed."}), 503
    if not all([compression_retriever, llm, memory, cohere_native_client]):
         logging.error("One or more essential chatbot components are None.")
         return jsonify({"error": "Chatbot components missing. Cannot process request."}), 500

    try:
        data = request.get_json()
        user_message = data.get('message')
        if not user_message or not user_message.strip():
            return jsonify({"error": "No message provided"}), 400
        logging.info(f"Received user message: '{user_message}'")

        # --- RAG Pipeline Steps ---
        retrieved_docs = compression_retriever.invoke(user_message)
        logging.info(f"Retrieved {len(retrieved_docs)} documents after reranking.")
        # Format context prioritizing images from rank 1 doc only, with clearer labels
        multimodal_context_parts = format_context_for_aya_vision(retrieved_docs)
        chat_history_messages = memory.load_memory_variables({})['chat_history']
        logging.debug(f"Loaded {len(chat_history_messages)} messages from chat history.")

        messages_for_llm_lc: List[BaseMessage] = []
        # --- REFINED SYSTEM PROMPT ---
        messages_for_llm_lc.append(SystemMessage(content="You are an AI assistant analyzing SolidCAM technical documentation. Your task is to answer the user's query accurately by synthesizing information from the provided text snippets and associated images (if any). Prioritize information from the highest-ranked source document (Document 1). Clearly state if the answer comes directly from visual inspection of an image or from the text. Be precise."))
        # --- END SYSTEM PROMPT ---
        messages_for_llm_lc.extend(chat_history_messages)
        current_human_message_content_list = [{"type": "text", "text": user_message}]
        current_human_message_content_list.extend(multimodal_context_parts) # Context now has clearer labels within text parts
        messages_for_llm_lc.append(HumanMessage(content=current_human_message_content_list))

        bot_answer = "Sorry, I encountered an issue generating a response."

        # --- Invoke LLM (Try Langchain, fallback to Native SDK) ---
        try:
            logging.info(f"Attempting LLM invocation via Langchain ChatCohere wrapper (Model: {COHERE_GEN_MODEL})...")
            ai_response = llm.invoke(messages_for_llm_lc)
            bot_answer = ai_response.content if hasattr(ai_response, 'content') else bot_answer
            logging.info("LLM invocation via Langchain wrapper successful.")
        except Exception as langchain_llm_error:
            logging.warning(f"Error invoking LLM with Langchain wrapper: {langchain_llm_error}", exc_info=True)
            logging.warning("Falling back to native Cohere SDK.")
            try:
                logging.info("Attempting LLM invocation via native Cohere SDK...")
                native_sdk_messages = convert_langchain_messages_to_cohere_format(messages_for_llm_lc)
                native_response = cohere_native_client.chat(
                    model=COHERE_GEN_MODEL, messages=native_sdk_messages, temperature=0.2
                )
                bot_answer = native_response.text if hasattr(native_response, 'text') else bot_answer
                logging.info("LLM invocation via native Cohere SDK successful.")
            except Exception as native_sdk_error:
                logging.error(f"Error invoking LLM with native Cohere SDK: {native_sdk_error}", exc_info=True)
                error_detail = str(native_sdk_error)
                bot_answer = f"Error processing request with the language model (Native SDK fallback failed: {error_detail[:100]}...)."

        # --- Save context and prepare response ---
        memory.save_context({"question": user_message}, {"answer": bot_answer})
        logging.debug("Saved text context to memory.")

        source_documents_data = []
        for doc_obj in retrieved_docs: # Use the initially retrieved docs for source info
            metadata = doc_obj.metadata if hasattr(doc_obj, 'metadata') else {}
            page_content_str = doc_obj.page_content if hasattr(doc_obj, 'page_content') else ""
            image_urls_for_display = []
            page_num_src = metadata.get('page')
            img_ids_src = metadata.get('image_ids', [])
            if page_num_src is not None and img_ids_src and metadata.get('has_images'):
                 image_urls_for_display = [get_image_data_url(page_num_src, img_id)
                                           for img_id in img_ids_src if get_image_data_url(page_num_src, img_id)]
            source_documents_data.append({
                "id": metadata.get('vector_id', 'N/A'), "page": metadata.get('page', 'N/A'),
                "header": metadata.get('header', 'N/A'), "snippet": metadata.get('text_snippet', page_content_str)[:300] + "...",
                "image_ids": metadata.get('image_ids', []), "has_images": metadata.get('has_images', False),
                "image_data_urls": image_urls_for_display
            })
        
        logging.info(f"Sending bot answer: '{bot_answer}' with {len(source_documents_data)} sources.")
        return jsonify({"answer": bot_answer, "sources": source_documents_data})

    except Exception as e:
        logging.error(f"Error processing chat message in /chat endpoint: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred while processing your message."}), 500

if __name__ == '__main__':
    if not components_initialized:
        logging.warning("Chatbot components failed to initialize on startup. The app might not function correctly.")
    app.run(debug=True, host='0.0.0.0', port=7001)
