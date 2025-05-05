# query_script.py (Final Version with Rerank & Enhanced Display)

import os
import logging
import cohere
from pinecone import Pinecone
from dotenv import load_dotenv
import time
import sys

# --- Configuration ---
load_dotenv() # Load environment variables from .env file
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API Keys and Environment Variables ---
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# --- Pinecone Settings ---
PINECONE_INDEX_NAME = "solidcam-chatbot-image-embeddings"
# Dimension must match the index created in Pinecone and Cohere model output
PINECONE_DIMENSION = 1024

# --- Cohere Settings ---
COHERE_MODEL = 'embed-v4.0' # For embedding queries
COHERE_INPUT_TYPE_QUERY = "search_query" # Correct type for retrieval queries
COHERE_OUTPUT_DIMENSION = 1024 # Explicitly match Pinecone index dimension
COHERE_RERANK_MODEL = 'rerank-english-v3.0' # Or 'rerank-multilingual-v3.0' if needed

# --- Query Settings ---
INITIAL_TOP_K = 10 # Retrieve more initial candidates from Pinecone for reranking
FINAL_TOP_N = 3    # Show the top N results after reranking

# --- Helper Function for Querying Pinecone ---
def query_pinecone(co: cohere.ClientV2, pc_index: Pinecone.Index, query_text: str, top_k: int):
    """
    Embeds a query using Cohere (v4) and fetches initial candidates from Pinecone.
    Returns the list of matches or None on error.
    """
    logging.info(f"Embedding query for Pinecone: '{query_text[:100]}...'")
    try:
        # Embed the query using Cohere ClientV2
        response = co.embed(
            texts=[query_text],
            model=COHERE_MODEL,
            input_type=COHERE_INPUT_TYPE_QUERY,
            embedding_types=["float"],
            output_dimension=COHERE_OUTPUT_DIMENSION # Explicitly set dimension
        )
        query_embedding = response.embeddings.float[0]
        logging.info(f"Query embedded successfully (Dim: {len(query_embedding)}).")

        # Query Pinecone
        logging.info(f"Querying Pinecone index '{PINECONE_INDEX_NAME}' with top_k={top_k}...")
        query_results = pc_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True # Crucial to get metadata back
        )
        logging.info("Pinecone query successful.")
        # Return the matches list directly
        return query_results.matches if hasattr(query_results, 'matches') else []

    except Exception as e:
        logging.error(f"Error during query embedding or Pinecone search: {e}", exc_info=True)
        return None

# --- Helper Function for Reranking ---
def rerank_results(co: cohere.ClientV2, query_text: str, pinecone_matches: list, top_n: int):
    """
    Reranks Pinecone matches using Cohere Rerank.
    Returns a new list of reranked matches (preserving original metadata).
    """
    if not pinecone_matches:
        return []

    # Extract text snippets for reranking. Using the longer snippet stored now.
    documents_to_rerank = []
    valid_original_matches = []
    for match in pinecone_matches:
        # Ensure metadata and the specific key exist before accessing
        if hasattr(match, 'metadata') and match.metadata and 'text_snippet' in match.metadata:
             documents_to_rerank.append(match.metadata['text_snippet'])
             valid_original_matches.append(match) # Keep track of the original match object
        else:
            logging.warning(f"Skipping match ID {match.id} for reranking due to missing metadata or text_snippet.")


    if not documents_to_rerank:
        logging.warning("No valid text snippets found in Pinecone results for reranking.")
        return []

    logging.info(f"Reranking {len(documents_to_rerank)} candidates using '{COHERE_RERANK_MODEL}'...")
    try:
        # Call the Cohere Rerank API
        rerank_response = co.rerank(
            model=COHERE_RERANK_MODEL,
            query=query_text,
            documents=documents_to_rerank,
            top_n=top_n # Ask the API to return only the top N results
        )
        logging.info("Cohere rerank successful.")

        # Map reranked results back to original Pinecone match data
        reranked_matches = []
        if hasattr(rerank_response, 'results'):
            for rerank_result in rerank_response.results:
                # The index from rerank corresponds to the index in the documents list sent
                original_index = rerank_result.index
                # Check if the original_index is valid for the list we actually sent
                if 0 <= original_index < len(valid_original_matches):
                    # Get the corresponding original match data using the index
                    original_match = valid_original_matches[original_index]
                    # Create a new dictionary holding the reranked info + original metadata
                    reranked_match_data = {
                        "id": original_match.id,
                        "score": rerank_result.relevance_score, # Use the new relevance score
                        "metadata": original_match.metadata # Keep the original metadata
                    }
                    reranked_matches.append(reranked_match_data)
                else:
                    logging.error(f"Rerank result index {original_index} is out of bounds for valid original matches list (size {len(valid_original_matches)}).")
            return reranked_matches
        else:
            logging.error("Rerank response did not contain 'results'. Response: {rerank_response}")
            return []

    except Exception as e:
        logging.error(f"Error during Cohere rerank: {e}", exc_info=True)
        return [] # Return empty list on rerank error

# --- Helper Function for Displaying Results (Handles Reranked Output) ---
def display_results(query_text: str, results_list: list):
    """
    Prints the query and its results (expects list of dicts from rerank_results).
    """
    print("-" * 80)
    print(f"QUERY: \"{query_text}\"")
    print("-" * 80)

    if not results_list:
        print(">>> No results found or error during processing.")
        print("-" * 80) # Add closing separator even if no results
        return

    print(f">>> Found {len(results_list)} results (Top {len(results_list)} after reranking):")
    for i, match_data in enumerate(results_list):
        # Extract data from the dictionary structure created by rerank_results
        match_id = match_data.get('id', 'N/A')
        score = match_data.get('score', 0.0)
        metadata = match_data.get('metadata', None)

        print(f"\n  Result #{i+1} (Relevance Score: {score:.4f})") # Show rerank score
        print(f"  ID: {match_id}") # Now shows readable ID like pageX-chunkY
        if metadata:
            # Safely get metadata fields using .get() with default values
            print(f"  Metadata:")
            print(f"    Page: {metadata.get('page', 'N/A')}")
            print(f"    Header: {metadata.get('header', 'N/A')}")
            print(f"    Has Images: {metadata.get('has_images', 'N/A')}")
            print(f"    Image IDs: {metadata.get('image_ids', 'N/A')}")
            print(f"    Text Snippet (up to 1000 chars): \"{metadata.get('text_snippet', 'N/A')}...\"")
        else:
            print("  Metadata: None")
    print("-" * 80)

# --- Main Execution Block ---
if __name__ == "__main__":
    main_start_time = time.time()
    logging.info("===== Starting Query Script (with Reranking, Updated Display) =====")

    # --- Initialize Clients ---
    co = None # Initialize to None
    index = None # Initialize to None
    try:
        logging.info("Initializing Cohere ClientV2...")
        co = cohere.ClientV2(api_key=COHERE_API_KEY)
        logging.info("Cohere ClientV2 initialized.")

        logging.info("Initializing Pinecone client...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        logging.info(f"Connecting to Pinecone index '{PINECONE_INDEX_NAME}'...")
        # Check if index exists
        # Use list_indexes().get('indexes', []) for safety in case 'indexes' key is missing
        if PINECONE_INDEX_NAME not in [idx.get('name') for idx in pc.list_indexes().get('indexes', []) if idx.get('name')]:
             logging.error(f"Pinecone index '{PINECONE_INDEX_NAME}' not found!")
             sys.exit(f"Error: Index '{PINECONE_INDEX_NAME}' does not exist.")
        index = pc.Index(PINECONE_INDEX_NAME) # Get the index object
        index_stats = index.describe_index_stats()
        logging.info(f"Connected to index. Stats: {index_stats}")
        # Verify index dimension matches expectation
        if index_stats.get('dimension') != PINECONE_DIMENSION:
             logging.warning(f"Script expects dimension {PINECONE_DIMENSION}, but index reports {index_stats.get('dimension')}. Check configuration.")

    except Exception as e:
        logging.error(f"Failed to initialize clients or connect to index: {e}", exc_info=True)
        sys.exit(1) # Exit if clients cannot be initialized

    # --- Define Test Queries ---
    test_queries = [
        "What operations are shown in the image for creating a threaded countersink hole?",
        "What is a Default set described near the M6/M8 diagram?",
        "How is a Machining Process Table defined?",
        "What are the parameters required for the threaded hole example shown in the diagram?",
        "How can I copy a Machining Process using the manager?",
        "Show the formula for centering depth calculation.",
        "What fields are in the Machining Process Table Manager?",
        "Show the 'Add machine' dialog box.",
        "What does the 'G' column represent in the Used Parameters Table?",
        "What menu option is available for 'Other Parameters'?",
    ]

    # --- Execute Queries ---
    logging.info("===== Starting Test Queries (with Reranking) =====")
    if not test_queries:
         logging.warning("No test queries defined in the script.")
    else:
        # Check if clients were successfully initialized before looping
        if co is None or index is None:
             logging.error("API clients not initialized. Cannot run queries.")
        else:
            for i, query in enumerate(test_queries):
                print(f"\n===== Running Test Query #{i+1} =====")
                # 1. Get initial candidates from Pinecone
                initial_matches = query_pinecone(co=co, pc_index=index, query_text=query, top_k=INITIAL_TOP_K)

                # 2. Rerank the initial candidates
                if initial_matches is not None: # Check for errors in query_pinecone
                    reranked_matches = rerank_results(co=co, query_text=query, pinecone_matches=initial_matches, top_n=FINAL_TOP_N)
                    # 3. Display reranked results
                    display_results(query_text=query, results_list=reranked_matches)
                else:
                    # Handle case where initial query failed
                    display_results(query_text=query, results_list=[])

                time.sleep(1.5) # Increase delay slightly for rerank API call
    logging.info("===== Finished Test Queries =====")

    # --- Script Completion ---
    logging.info("Query script finished successfully.")
    main_end_time = time.time()
    logging.info(f"Total query script execution time: {main_end_time - main_start_time:.2f} seconds")

