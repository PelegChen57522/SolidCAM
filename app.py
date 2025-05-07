# app.py
# Flask web application for the RAG chatbot.
# Includes logic to pass image data URLs to the frontend.

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
# from langchain_core.documents import Document # Not explicitly used here but good for reference

# --- Configuration ---
load_dotenv() # Load environment variables from .env file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API Keys ---
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not COHERE_API_KEY:
    logging.critical("COHERE_API_KEY not found. Flask app may not function correctly.")
if not PINECONE_API_KEY:
    logging.critical("PINECONE_API_KEY not found. Flask app may not function correctly.")

# --- Pinecone Settings ---
PINECONE_INDEX_NAME = "solidcam-chatbot-image-embeddings"

# --- Cohere Settings ---
COHERE_EMBED_MODEL_NAME = 'embed-v4.0'
COHERE_TARGET_EMBED_DIMENSION = 1024
COHERE_GEN_MODEL = 'command-r' # Or 'command-r-plus' if available and preferred
COHERE_RERANK_MODEL = 'rerank-english-v3.0' # Or other rerank models
# Parameters for ClientV2().embed() used in custom embedding class
COHERE_INPUT_TYPE_DOC_FOR_V2_EMBED = "search_document"
COHERE_INPUT_TYPE_QUERY_FOR_V2_EMBED = "search_query"
COHERE_EMBEDDING_TYPES_FOR_V2_EMBED = ["float"]

# --- RAG Settings ---
RERANK_TOP_N = 3
INITIAL_RETRIEVAL_K = 10

# --- Custom Langchain-compatible Cohere Embeddings Class using ClientV2 ---
class CustomCohereEmbeddingsWithClientV2(Embeddings):
    """
    Custom Langchain Embeddings class using cohere.ClientV2.
    Ensures `output_dimension` is respected for query embeddings to match document embeddings.
    Uses the `inputs` parameter structure for Cohere's embed API v4.0.
    """
    client: cohere.ClientV2
    model: str
    output_dimension: int
    embedding_types: List[str]
    input_type: str # To specify "search_query" or "search_document"

    def __init__(self, api_key: str, model_name: str, output_dim: int, embedding_types: List[str], input_type_for_embedding: str):
        super().__init__()
        self.client = cohere.ClientV2(api_key=api_key)
        self.model = model_name
        self.output_dimension = output_dim
        self.embedding_types = embedding_types
        self.input_type = input_type_for_embedding

    def _prepare_inputs_for_clientv2(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Prepares the structured input for ClientV2 embed method's 'inputs' parameter."""
        structured_inputs = []
        for text_item in texts:
            if text_item and text_item.strip():
                # For embed-v4.0, the `inputs` parameter takes a list of dictionaries,
                # where each dictionary has a "content" key.
                # The "content" is a list of parts (e.g., text, image_url).
                # For query embedding, we typically only have text.
                content_list = [{"type": "text", "text": text_item.strip()}]
                structured_inputs.append({"content": content_list})
            else:
                logging.debug(f"Skipping empty text item during input preparation: '{text_item}'")
        return structured_inputs

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents. This will use 'search_document' input type."""
        if not texts: return [[] for _ in texts] # Return list of empty lists if no texts
        
        # Override input_type for document embedding if it was set for query
        original_input_type = self.input_type
        self.input_type = COHERE_INPUT_TYPE_DOC_FOR_V2_EMBED
        
        prepared_inputs = self._prepare_inputs_for_clientv2(texts)
        
        self.input_type = original_input_type # Restore original input type

        if not prepared_inputs:
            logging.warning("embed_documents: All input texts were empty. Returning list of empty embeddings.")
            return [[] for _ in texts]

        try:
            response = self.client.embed(
                model=self.model,
                inputs=prepared_inputs,
                input_type=COHERE_INPUT_TYPE_DOC_FOR_V2_EMBED, # Explicitly for documents
                embedding_types=self.embedding_types,
                output_dimension=self.output_dimension
            )
            
            embeddings = None
            if hasattr(response, 'embeddings'):
                if isinstance(response.embeddings, list): embeddings = response.embeddings
                elif hasattr(response.embeddings, 'float') and isinstance(response.embeddings.float, list): embeddings = response.embeddings.float
                elif isinstance(response.embeddings, dict) and 'float' in response.embeddings: embeddings = response.embeddings['float']

            if embeddings and len(embeddings) == len(prepared_inputs):
                # Map results back to the original length of `texts`
                valid_texts_map = {text: emb for text, emb in zip([t for t in texts if t and t.strip()], embeddings)}
                final_embeddings = [valid_texts_map.get(text, []) for text in texts]
                return final_embeddings
            else:
                logging.error(f"Cohere embed_documents error or mismatch. Expected {len(prepared_inputs)} embeddings. Got: {len(embeddings) if embeddings else 'None'}")
                return [[] for _ in texts]
        except Exception as e:
            logging.error(f"Error embedding documents with Cohere (ClientV2): {e}", exc_info=True)
            return [[] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query. This will use 'search_query' input type."""
        if not text or not text.strip():
            logging.warning("Embed_query received an empty query. Returning empty list.")
            return []

        prepared_inputs = self._prepare_inputs_for_clientv2([text])
        if not prepared_inputs: return []

        try:
            response = self.client.embed(
                model=self.model,
                inputs=prepared_inputs,
                input_type=COHERE_INPUT_TYPE_QUERY_FOR_V2_EMBED, # Explicitly for queries
                embedding_types=self.embedding_types,
                output_dimension=self.output_dimension
            )
            embeddings = None
            if hasattr(response, 'embeddings'):
                if isinstance(response.embeddings, list) and len(response.embeddings) > 0 : embeddings = response.embeddings[0] # Expect single list for single query
                elif hasattr(response.embeddings, 'float') and isinstance(response.embeddings.float, list) and len(response.embeddings.float) > 0: embeddings = response.embeddings.float[0]
                elif isinstance(response.embeddings, dict) and 'float' in response.embeddings and len(response.embeddings['float']) > 0 : embeddings = response.embeddings['float'][0]

            if embeddings:
                return embeddings
            else:
                logging.error(f"Cohere embed_query error or mismatch. Response: {response}")
                return []
        except Exception as e:
            logging.error(f"Error embedding query with Cohere (ClientV2): {e}", exc_info=True)
            return []

# --- Global Chatbot Chain Initialization ---
qa_chain = None

def initialize_chatbot():
    """Initializes the Langchain ConversationalRetrievalChain."""
    global qa_chain
    if not COHERE_API_KEY or not PINECONE_API_KEY:
        logging.error("API keys for Cohere or Pinecone are missing. Chatbot cannot be initialized.")
        return

    try:
        logging.info(f"Initializing CustomCohereEmbeddings for queries: model={COHERE_EMBED_MODEL_NAME}, dim={COHERE_TARGET_EMBED_DIMENSION}")
        # This instance is specifically for query embeddings
        custom_query_embeddings = CustomCohereEmbeddingsWithClientV2(
            api_key=COHERE_API_KEY,
            model_name=COHERE_EMBED_MODEL_NAME,
            output_dim=COHERE_TARGET_EMBED_DIMENSION,
            embedding_types=COHERE_EMBEDDING_TYPES_FOR_V2_EMBED,
            input_type_for_embedding=COHERE_INPUT_TYPE_QUERY_FOR_V2_EMBED # For queries
        )

        logging.info(f"Connecting to Pinecone index: {PINECONE_INDEX_NAME}")
        # PineconeVectorStore will use the `embed_query` method of custom_query_embeddings for similarity search
        # and `embed_documents` if it needs to embed documents (e.g. for from_texts, not used here)
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=custom_query_embeddings, # Pass the custom embeddings object
            text_key="text_snippet" # Assumes 'text_snippet' in metadata contains the main text
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

        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )
        logging.info("Conversation memory initialized.")

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=compression_retriever,
            memory=memory,
            return_source_documents=True,
            output_key='answer'
        )
        logging.info("ConversationalRetrievalChain created successfully and is ready.")
    except Exception as e:
        logging.error(f"Fatal error during Langchain component initialization: {e}", exc_info=True)
        qa_chain = None

# --- Flask App Setup ---
app = Flask(__name__, template_folder='templates') # Ensure 'templates' folder exists for index.html
CORS(app)

initialize_chatbot() # Initialize chatbot components when app starts

@app.route('/')
def index_route():
    """Serves the main HTML page for the chatbot UI."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat_route():
    """Handles chat messages from the user and returns bot responses."""
    global qa_chain
    if qa_chain is None:
        logging.error("Chatbot chain is not initialized. Attempting re-initialization.")
        initialize_chatbot() # Attempt to re-initialize
        if qa_chain is None:
             logging.error("Re-initialization failed. Chatbot unavailable.")
             return jsonify({"error": "Chatbot is not available. Initialization failed. Please check server logs."}), 503

    try:
        data = request.get_json()
        user_message = data.get('message')

        if not user_message or not user_message.strip():
            return jsonify({"error": "No message provided or message is empty"}), 400

        logging.info(f"Received user message: '{user_message}'")
        
        # Invoke the Langchain chain
        result = qa_chain.invoke({"question": user_message})
        bot_answer = result.get('answer', "Sorry, I couldn't generate a response at this moment.")
        
        source_documents_data = []
        if 'source_documents' in result and result['source_documents']:
            for doc_obj in result['source_documents']:
                metadata = doc_obj.metadata if hasattr(doc_obj, 'metadata') else {}
                page_content_str = doc_obj.page_content if hasattr(doc_obj, 'page_content') and doc_obj.page_content is not None else ""
                
                # Extract image_data_urls from metadata if present
                image_urls = metadata.get('image_data_urls', []) 
                
                source_documents_data.append({
                    "id": metadata.get('vector_id', 'N/A'),
                    "page": metadata.get('page', 'N/A'),
                    "header": metadata.get('header', 'N/A'),
                    "snippet": metadata.get('text_snippet', page_content_str)[:300] + "...", # Slightly longer snippet
                    "image_data_urls": image_urls # Pass the image URLs to the frontend
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
    # For production, use a WSGI server like Gunicorn.
    # Example: gunicorn -w 4 -b 0.0.0.0:7001 app:app
    # The host='0.0.0.0' makes the server accessible from your network.
    # Ensure the 'templates' folder with 'index.html' is in the same directory as app.py or adjust template_folder path.
    app.run(debug=True, host='0.0.0.0', port=7001) # debug=True for development
