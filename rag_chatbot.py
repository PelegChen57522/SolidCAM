# rag_chatbot.py

import os
import logging
from dotenv import load_dotenv
from typing import List
import sys

import cohere # For cohere.ClientV2
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
    logging.error("COHERE_API_KEY not found in environment variables.")
    sys.exit("Error: Missing COHERE_API_KEY. Please set it in your .env file.")
if not PINECONE_API_KEY:
    logging.error("PINECONE_API_KEY not found in environment variables.")
    sys.exit("Error: Missing PINECONE_API_KEY. Please set it in your .env file.")

# --- Pinecone Settings ---
PINECONE_INDEX_NAME = "solidcam-chatbot-image-embeddings"

# --- Cohere Settings ---
COHERE_EMBED_MODEL_NAME = 'embed-v4.0'
COHERE_TARGET_EMBED_DIMENSION = 1024
COHERE_GEN_MODEL = 'command-r'
COHERE_RERANK_MODEL = 'rerank-english-v3.0'
# Parameters for ClientV2().embed()
COHERE_INPUT_TYPE_DOC_FOR_V2_EMBED = "search_document"
COHERE_INPUT_TYPE_QUERY_FOR_V2_EMBED = "search_query"
COHERE_EMBEDDING_TYPES_FOR_V2_EMBED = ["float"]

# --- RAG Settings ---
RERANK_TOP_N = 3
INITIAL_RETRIEVAL_K = 10

# --- Custom Langchain-compatible Cohere Embeddings Class using ClientV2 ---
class CustomCohereEmbeddingsWithClientV2(Embeddings):
    """
    Custom Langchain Embeddings class that uses cohere.ClientV2
    and the 'inputs' parameter structure to ensure output_dimension is respected,
    mirroring the successful approach in embed_and_store.py.
    """
    client: cohere.ClientV2 # Use ClientV2
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
                # If the text is empty, we might need to add a placeholder or handle it
                # For now, let's skip empty texts to avoid sending empty content to API
                logging.debug(f"Skipping empty text item: '{text_item}'")
        return structured_inputs

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if not texts:
            return []
        
        prepared_inputs = self._prepare_inputs_for_clientv2(texts)
        
        # If all texts were empty/whitespace and prepared_inputs is empty
        if not prepared_inputs:
            logging.warning("embed_documents: All input texts were empty or whitespace. Returning list of empty embeddings.")
            return [[] for _ in texts] # Match original length with empty embeddings

        try:
            response = self.client.embed(
                model=self.model,
                inputs=prepared_inputs, # Use the structured inputs
                input_type=COHERE_INPUT_TYPE_DOC_FOR_V2_EMBED,
                embedding_types=self.embedding_types,
                output_dimension=self.output_dimension
            )
            if hasattr(response, 'embeddings') and response.embeddings and \
               hasattr(response.embeddings, 'float') and \
               len(response.embeddings.float) == len(prepared_inputs):
                
                # Map results back to the original length of `texts`, inserting [] for those that were filtered out
                embeddings_dict = {original_text: emb for original_text, emb in zip([t for t in texts if t and t.strip()], response.embeddings.float)}
                final_embeddings = [embeddings_dict.get(text, []) for text in texts]
                return final_embeddings
            else:
                num_embeddings_received = len(response.embeddings.float) if hasattr(response, 'embeddings') and hasattr(response.embeddings, 'float') else 'N/A'
                logging.error(f"Cohere (ClientV2) embed_documents response error or mismatch. Expected {len(prepared_inputs)} embeddings, got {num_embeddings_received}.")
                return [[] for _ in texts]
        except Exception as e:
            logging.error(f"Error embedding documents with Cohere (ClientV2): {e}", exc_info=True)
            return [[] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        if not text or not text.strip():
            logging.warning("Embed_query received an empty or whitespace-only query.")
            return []

        prepared_inputs = self._prepare_inputs_for_clientv2([text])
        if not prepared_inputs: # Should only happen if text was empty, handled above
             return []

        try:
            response = self.client.embed(
                model=self.model,
                inputs=prepared_inputs, # Use the structured inputs
                input_type=COHERE_INPUT_TYPE_QUERY_FOR_V2_EMBED,
                embedding_types=self.embedding_types,
                output_dimension=self.output_dimension
            )
            if hasattr(response, 'embeddings') and response.embeddings and \
               hasattr(response.embeddings, 'float') and \
               len(response.embeddings.float) == 1:
                return response.embeddings.float[0]
            else:
                logging.error(f"Cohere (ClientV2) embed_query response error or mismatch. Response: {response}")
                return []
        except Exception as e:
            logging.error(f"Error embedding query with Cohere (ClientV2): {e}", exc_info=True)
            return []

def main():
    logging.info("Initializing RAG Chatbot components...")
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
            text_key="text_snippet" # Use the metadata field containing the main text
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
        logging.info("ConversationalRetrievalChain created successfully.")

    except Exception as e:
        logging.error(f"Fatal error during Langchain component initialization: {e}", exc_info=True)
        print("Error: Could not initialize chatbot components. Please check API keys and configurations. Exiting.")
        return

    print("\n--- SolidCAM RAG Chatbot ---")
    print(f"Powered by Cohere ({COHERE_GEN_MODEL}) and Pinecone.")
    print(f"Embedding with {COHERE_EMBED_MODEL_NAME} at {COHERE_TARGET_EMBED_DIMENSION} dimensions (using ClientV2 for queries).")
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

            logging.info(f"Processing user query: '{user_query}'")
            result = qa_chain.invoke({"question": user_query})

            print(f"\nBot: {result['answer']}")

            if 'source_documents' in result and result['source_documents']:
                print("\n--- Retrieved Sources (Post-Reranking) ---")
                for i, doc in enumerate(result['source_documents']):
                    metadata = doc.metadata
                    print(f"  Source {i+1}: ID='{metadata.get('id', 'N/A')}', "
                          f"Page={metadata.get('page', 'N/A')}, "
                          f"Header='{metadata.get('header', 'N/A')}'")
                print("--- End Sources ---")
        except Exception as e:
            logging.error(f"Error during chat interaction: {e}", exc_info=True)
            print("\nBot: My apologies, I encountered an issue while processing your request. Please try again.")

if __name__ == "__main__":
    main()
