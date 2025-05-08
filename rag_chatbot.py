import os
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any
import sys

import cohere
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_cohere import CohereRerank, ChatCohere
from langchain.retrievers import ContextualCompressionRetriever

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API Keys ---
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not COHERE_API_KEY:
    logging.error("COHERE_API_KEY not found. Please set it in your .env file.")
    sys.exit("Error: Missing COHERE_API_KEY.")
if not PINECONE_API_KEY:
    logging.error("PINECONE_API_KEY not found. Please set it in your .env file.")
    sys.exit("Error: Missing PINECONE_API_KEY.")

# --- Pinecone Settings ---
PINECONE_INDEX_NAME = "solidcam-chatbot-image-embeddings"

# --- Cohere Settings ---
COHERE_EMBED_MODEL_NAME = 'embed-v4.0'
COHERE_TARGET_EMBED_DIMENSION = 1024
COHERE_GEN_MODEL = 'command-r'
COHERE_RERANK_MODEL = 'rerank-english-v3.0'
COHERE_INPUT_TYPE_DOC_FOR_V2_EMBED = "search_document"
COHERE_INPUT_TYPE_QUERY_FOR_V2_EMBED = "search_query"
COHERE_EMBEDDING_TYPES_FOR_V2_EMBED = ["float"]

# --- RAG Settings ---
RERANK_TOP_N = 3
INITIAL_RETRIEVAL_K = 10

# --- Custom Langchain-compatible Cohere Embeddings Class (same as in app.py) ---
class CustomCohereEmbeddingsWithClientV2(Embeddings):
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
            else:
                logging.debug(f"Skipping empty text item: '{text_item}'")
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
            logging.error(f"Error embedding documents (CLI): {e}", exc_info=True)
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
            logging.error(f"Error embedding query (CLI): {e}", exc_info=True)
            return []

def main():
    logging.info("Initializing RAG Chatbot components (CLI)...")
    qa_chain_cli = None
    try:
        custom_query_embeddings_cli = CustomCohereEmbeddingsWithClientV2(
            api_key=COHERE_API_KEY, model_name=COHERE_EMBED_MODEL_NAME,
            output_dim=COHERE_TARGET_EMBED_DIMENSION, embedding_types=COHERE_EMBEDDING_TYPES_FOR_V2_EMBED,
            input_type_for_embedding=COHERE_INPUT_TYPE_QUERY_FOR_V2_EMBED
        )
        vectorstore_cli = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME, embedding=custom_query_embeddings_cli, text_key="text_snippet"
        )
        base_retriever_cli = vectorstore_cli.as_retriever(search_kwargs={"k": INITIAL_RETRIEVAL_K})
        reranker_cli = CohereRerank(model=COHERE_RERANK_MODEL, top_n=RERANK_TOP_N, cohere_api_key=COHERE_API_KEY)
        compression_retriever_cli = ContextualCompressionRetriever(base_compressor=reranker_cli, base_retriever=base_retriever_cli)
        llm_cli = ChatCohere(model=COHERE_GEN_MODEL, temperature=0.2, cohere_api_key=COHERE_API_KEY)
        memory_cli = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
        
        qa_chain_cli = ConversationalRetrievalChain.from_llm(
            llm=llm_cli, retriever=compression_retriever_cli, memory=memory_cli,
            return_source_documents=True, output_key='answer'
        )
        logging.info("CLI Chatbot components initialized successfully.")
    except Exception as e:
        logging.error(f"Fatal error during CLI Langchain component initialization: {e}", exc_info=True)
        print("\nError: Could not initialize chatbot components. Please check API keys and configurations. Exiting.")
        return

    print("\n--- SolidCAM RAG Chatbot (CLI) ---")
    print(f"Powered by Cohere ({COHERE_GEN_MODEL}) and Pinecone.")
    print(f"Embedding with {COHERE_EMBED_MODEL_NAME} (multimodal for docs, text for queries).")
    print("Ask questions about the 'Milling 2024 Machining Processes' document.")
    print("Type 'quit' or 'exit' to end the chat.")

    while True:
        try:
            user_query = input("\nYou: ")
            if user_query.lower() in ["quit", "exit"]:
                print("Exiting chatbot.")
                break
            if not user_query.strip():
                continue

            logging.info(f"CLI - Processing user query: '{user_query}'")
            result = qa_chain_cli.invoke({"question": user_query})

            print(f"\nBot: {result.get('answer', 'No answer generated.')}")

            if 'source_documents' in result and result['source_documents']:
                print("\n--- Retrieved Sources (Post-Reranking) ---")
                for i, doc in enumerate(result['source_documents']):
                    metadata = doc.metadata
                    print(f"  Source {i+1}: ID='{metadata.get('vector_id', 'N/A')}', "
                          f"Page={metadata.get('page', 'N/A')}, "
                          f"Header='{metadata.get('header', 'N/A')}', "
                          f"Has Images={metadata.get('has_images', False)}")
                    # For CLI, we don't display images, but we can acknowledge if they are present.
                print("--- End Sources ---")
        except Exception as e:
            logging.error(f"Error during CLI chat interaction: {e}", exc_info=True)
            print("\nBot: My apologies, I encountered an issue. Please try again.")

if __name__ == "__main__":
    main()
