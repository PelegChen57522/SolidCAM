import os
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any # For type hinting
import sys

import cohere # For cohere.ClientV2 in custom embeddings
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS # For handling Cross-Origin Resource Sharing

# Langchain and Cohere specific imports
from langchain_core.embeddings import Embeddings # Base class for custom embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_cohere import CohereRerank, ChatCohere # Langchain's Cohere components
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document # To help structure source documents if needed

# --- Configuration ---
load_dotenv() # Load environment variables from .env file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API Keys ---
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Check for API keys at the start of the module
if not COHERE_API_KEY:
    logging.critical("COHERE_API_KEY not found in environment variables. Flask app will not function correctly.")
    # sys.exit("Error: Missing COHERE_API_KEY.") # Exiting here would prevent Flask from starting
if not PINECONE_API_KEY:
    logging.critical("PINECONE_API_KEY not found in environment variables. Flask app will not function correctly.")
    # sys.exit("Error: Missing PINECONE_API_KEY.")

# --- Pinecone Settings ---
PINECONE_INDEX_NAME = "solidcam-chatbot-image-embeddings" # Must match your populated index

# --- Cohere Settings ---
COHERE_EMBED_MODEL_NAME = 'embed-v4.0'        # Model name for embeddings
COHERE_TARGET_EMBED_DIMENSION = 1024          # Desired dimension for Pinecone and queries
COHERE_GEN_MODEL = 'command-r'                # For generating answers
COHERE_RERANK_MODEL = 'rerank-english-v3.0'   # For reranking retrieved documents
# Parameters for ClientV2().embed() used in custom embedding class
COHERE_INPUT_TYPE_DOC_FOR_V2_EMBED = "search_document"
COHERE_INPUT_TYPE_QUERY_FOR_V2_EMBED = "search_query"
COHERE_EMBEDDING_TYPES_FOR_V2_EMBED = ["float"]

# --- RAG Settings ---
RERANK_TOP_N = 3          # Number of documents to pass to the LLM after reranking
INITIAL_RETRIEVAL_K = 10  # Number of initial documents to fetch from Pinecone

# --- Custom Langchain-compatible Cohere Embeddings Class using ClientV2 ---
# This class ensures output_dimension is respected, based on successful embed_and_store.py pattern
class CustomCohereEmbeddingsWithClientV2(Embeddings):
    client: cohere.ClientV2 # Use cohere.ClientV2
    model: str
    output_dimension: int
    embedding_types: List[str]

    def __init__(self, api_key: str, model_name: str, output_dim: int, embedding_types: List[str]):
        super().__init__()
        self.client = cohere.ClientV2(api_key=api_key) # Initialize ClientV2
        self.model = model_name
        self.output_dimension = output_dim
        self.embedding_types = embedding_types

    def _prepare_inputs_for_clientv2(self, texts: List[str]) -> List[dict]:
        """Prepares the structured input for ClientV2 embed method's 'inputs' parameter."""
        structured_inputs = []
        for text_item in texts:
            if text_item and text_item.strip(): # Ensure text_item is not empty or just whitespace
                # This structure worked in embed_and_store.py for ClientV2
                content_list = [{"type": "text", "text": text_item}]
                structured_inputs.append({"content": content_list})
            else:
                logging.debug(f"Skipping empty text item during input preparation: '{text_item}'")
        return structured_inputs

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if not texts: return []
        
        prepared_inputs = self._prepare_inputs_for_clientv2(texts)
        
        if not prepared_inputs: # If all texts were empty/whitespace
            logging.warning("embed_documents: All input texts were empty or whitespace. Returning list of empty embeddings.")
            return [[] for _ in texts] # Match original length with empty embeddings

        try:
            response = self.client.embed(
                model=self.model,
                inputs=prepared_inputs, # Use the structured inputs
                input_type=COHERE_INPUT_TYPE_DOC_FOR_V2_EMBED, # Use constant
                embedding_types=self.embedding_types,
                output_dimension=self.output_dimension
            )
            # Access embeddings from response.embeddings.float for ClientV2
            if hasattr(response, 'embeddings') and response.embeddings and \
               hasattr(response.embeddings, 'float') and \
               len(response.embeddings.float) == len(prepared_inputs):
                
                # Map results back to the original length of `texts`, inserting [] for those that were filtered out
                # This ensures the output list length matches the input list length, as Langchain expects.
                valid_input_texts = [t for t in texts if t and t.strip()] # Texts that were actually embedded
                embeddings_map = {original_text: emb for original_text, emb in zip(valid_input_texts, response.embeddings.float)}
                final_embeddings = [embeddings_map.get(text, []) for text in texts]
                return final_embeddings
            else:
                num_embeddings_received = len(response.embeddings.float) if hasattr(response, 'embeddings') and hasattr(response.embeddings, 'float') else 'N/A'
                logging.error(f"Cohere (ClientV2) embed_documents response error or mismatch. Expected {len(prepared_inputs)} embeddings, got {num_embeddings_received}.")
                return [[] for _ in texts] # Return empty embeddings for all on error
        except Exception as e:
            logging.error(f"Error embedding documents with Cohere (ClientV2): {e}", exc_info=True)
            return [[] for _ in texts] # Return empty embeddings for all on error

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        if not text or not text.strip(): # Handle empty or whitespace-only query
            logging.warning("Embed_query received an empty or whitespace-only query. Returning empty list.")
            return []

        prepared_inputs = self._prepare_inputs_for_clientv2([text]) # Prepare input for the single query
        if not prepared_inputs: # Should only happen if text was empty, handled above
             return []

        try:
            response = self.client.embed(
                model=self.model,
                inputs=prepared_inputs, # Use the structured inputs
                input_type=COHERE_INPUT_TYPE_QUERY_FOR_V2_EMBED, # Use constant
                embedding_types=self.embedding_types,
                output_dimension=self.output_dimension
            )
            if hasattr(response, 'embeddings') and response.embeddings and \
               hasattr(response.embeddings, 'float') and \
               len(response.embeddings.float) == 1: # Expect one embedding for a single query
                return response.embeddings.float[0]
            else:
                logging.error(f"Cohere (ClientV2) embed_query response error or mismatch. Response: {response}")
                return [] # Return empty list on error
        except Exception as e:
            logging.error(f"Error embedding query with Cohere (ClientV2): {e}", exc_info=True)
            return [] # Return empty list on error

# --- Global Chatbot Chain Initialization ---
qa_chain = None # Global variable for the Langchain chain

def initialize_chatbot():
    """Initializes the Langchain ConversationalRetrievalChain."""
    global qa_chain
    if not COHERE_API_KEY or not PINECONE_API_KEY: # Double check API keys before expensive init
        logging.error("API keys for Cohere or Pinecone are missing. Chatbot cannot be initialized.")
        return

    try:
        logging.info(f"Initializing CustomCohereEmbeddingsWithClientV2: model={COHERE_EMBED_MODEL_NAME}, dimension={COHERE_TARGET_EMBED_DIMENSION}")
        custom_embeddings = CustomCohereEmbeddingsWithClientV2(
            api_key=COHERE_API_KEY,
            model_name=COHERE_EMBED_MODEL_NAME,
            output_dim=COHERE_TARGET_EMBED_DIMENSION,
            embedding_types=COHERE_EMBEDDING_TYPES_FOR_V2_EMBED # Pass the constant
        )

        logging.info(f"Connecting to Pinecone index: {PINECONE_INDEX_NAME}")
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=custom_embeddings,
            text_key="text_snippet" # Use the metadata field containing the main text for page_content
        )
        logging.info("Successfully connected to Pinecone.")

        base_retriever = vectorstore.as_retriever(search_kwargs={"k": INITIAL_RETRIEVAL_K})
        logging.info(f"Base retriever initialized (fetches k={INITIAL_RETRIEVAL_K}).")

        logging.info(f"Initializing Cohere Reranker: {COHERE_RERANK_MODEL}")
        reranker = CohereRerank(model=COHERE_RERANK_MODEL, top_n=RERANK_TOP_N, cohere_api_key=COHERE_API_KEY)

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever
        )
        logging.info(f"Contextual Compression Retriever initialized (reranks to top_n={RERANK_TOP_N}).")

        logging.info(f"Initializing Cohere Chat model: {COHERE_GEN_MODEL}")
        llm = ChatCohere(model=COHERE_GEN_MODEL, temperature=0.2, cohere_api_key=COHERE_API_KEY)

        # For a web app, conversation memory needs careful management if you want separate
        # conversations per user/session. For this example, we use a single global memory
        # associated with the qa_chain. This means all users share the same conversation history.
        # For per-user history, you'd typically create/retrieve memory instances based on session IDs.
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer' # Ensures the LLM's response is stored as 'answer'
        )
        logging.info("Conversation memory initialized.")

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=compression_retriever,
            memory=memory, # The chain will use this memory instance
            return_source_documents=True, # To inspect which documents were used
            output_key='answer' # Consistent output key
        )
        logging.info("ConversationalRetrievalChain created successfully and is ready.")
    except Exception as e:
        logging.error(f"Fatal error during Langchain component initialization: {e}", exc_info=True)
        qa_chain = None # Ensure chain is None if initialization fails

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes, allowing frontend from any origin (useful for dev)

# Initialize the chatbot when the Flask app starts
# This will run once when the first request comes in or at startup depending on Flask version/config.
# For more control, you can use Flask's @app.before_first_request (deprecated) or other app context methods.
initialize_chatbot()

@app.route('/')
def index_route(): # Renamed from 'index' to avoid conflict with Pinecone's index object if it were global
    """Serves the main HTML page for the chatbot UI."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat_route(): # Renamed from 'chat' for clarity
    """Handles chat messages from the user and returns bot responses."""
    global qa_chain # Access the globally initialized chain
    if qa_chain is None:
        logging.error("Chatbot chain is not initialized. API keys might be missing or Pinecone/Cohere connection failed.")
        # Attempt to re-initialize if it failed earlier (e.g., due to transient network issue at startup)
        # This is a simple retry, more robust error handling might be needed in production.
        initialize_chatbot()
        if qa_chain is None: # If still not initialized
             return jsonify({"error": "Chatbot is not available. Initialization failed. Please check server logs."}), 503 # Service Unavailable

    try:
        data = request.get_json()
        user_message = data.get('message')

        if not user_message:
            return jsonify({"error": "No message provided in the request body"}), 400

        logging.info(f"Received user message via API: '{user_message}'")

        # Invoke the Langchain chain. The chain's internal memory handles conversation history.
        result = qa_chain.invoke({"question": user_message})

        bot_answer = result.get('answer', "Sorry, I couldn't generate a response at this moment.")
        
        source_documents_data = []
        if 'source_documents' in result and result['source_documents']:
            for doc_obj in result['source_documents']: # doc_obj is a Langchain Document object
                metadata = doc_obj.metadata if hasattr(doc_obj, 'metadata') else {}
                # Ensure page_content is a string, default to empty if None
                page_content_str = doc_obj.page_content if hasattr(doc_obj, 'page_content') and doc_obj.page_content is not None else ""
                
                source_documents_data.append({
                    "id": metadata.get('vector_id', 'N/A'), # Use 'vector_id' from metadata
                    "page": metadata.get('page', 'N/A'),
                    "header": metadata.get('header', 'N/A'),
                    "snippet": metadata.get('text_snippet', page_content_str)[:200] + "..." # Use text_snippet or page_content
                })
        
        logging.info(f"Sending bot answer: '{bot_answer}' with {len(source_documents_data)} sources.")
        return jsonify({
            "answer": bot_answer,
            "sources": source_documents_data
        })

    except Exception as e:
        logging.error(f"Error processing chat message in /chat endpoint: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred while processing your message."}), 500

if __name__ == '__main__':
    # For development, Flask's built-in server is fine.
    # For production, use a more robust WSGI server like Gunicorn or Waitress.
    # Example: gunicorn -w 4 -b 0.0.0.0:7001 app:app
    # The host='0.0.0.0' makes the server accessible from your network, not just localhost.
    app.run(debug=True, host='0.0.0.0', port=7001) # debug=True enables live reloading and debugger (for development only!)
