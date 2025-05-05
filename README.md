# SolidCAM Chatbot - Multimodal Embedding Workflow Test

## Project Goal

This project implements and tests a new workflow for generating multimodal embeddings (text and images) from PDF documents for the SolidCAM ChatBot. The goal was to leverage newer AI models (MistralAI OCR and Cohere Embed v4) capable of understanding both text and visual elements, addressing the limitation of previous workflows that excluded images.

This task was assigned by Ori Somekh (SolidCAM) to evaluate this approach using the provided `Milling 2024 Machining Processes.pdf` document. A key requirement was to **preserve the structural meaning** of the document, ensuring headers, subheaders, and their corresponding content (including images) remain associated for coherent retrieval. This was successfully achieved through a combination of refined header chunking (excluding non-semantic headers), Cohere's fused embedding, and Cohere's reranking capabilities.

## Final Workflow Overview

The implemented workflow consists of the following main steps:

1.  **PDF Processing (OCR):** The `MistralAI_OCR.py` script uploads the input PDF (`Milling 2024 Machining Processes.pdf`) to the Mistral AI API. It uses the `mistral-ocr-latest` model to extract structured text (Markdown) and associated image data (Base64 encoded) for each page. The output is saved as a structured JSON file (`processed_solidcam_doc.json`). This step requires a Mistral AI API key.
2.  **Chunking & Fused Embedding:** The `embed_and_store.py` script reads the `processed_solidcam_doc.json` file.
    * **Chunking:** It chunks the Markdown content based on H1, H2, and H3 headers, **excluding** specific non-content headers (like "See Also", "Related Topics") and list-like headers starting with `## - ` to improve semantic grouping.
    * **Image Association:** It identifies Markdown image tags within each chunk and associates the corresponding Base64 image data (formatted as Data URLs).
    * **Fused Embedding:** It uses the Cohere `embed-v4.0` model (via `ClientV2`) with the `inputs` parameter to create a single embedding vector for each chunk, combining the chunk's text and any associated image data URLs. This captures the text-image relationship.
    * **Storage:** The generated 1024-dimension vectors (with human-readable IDs like `page<P>-chunk<C>`) and associated metadata (1-based page, header, longer text snippet, image IDs) are uploaded to a specified Pinecone index. This step requires Cohere and Pinecone API keys.
3.  **Querying & Re-ranking:** The `query_script.py` script provides a way to test retrieval:
    * **Query Embedding:** Embeds a text query using Cohere `embed-v4.0` (`input_type="search_query"`).
    * **Initial Retrieval:** Fetches an initial set of candidate chunks (e.g., top 10) from Pinecone based on vector similarity.
    * **Re-ranking:** Uses Cohere's `rerank-english-v3.0` model to re-order the initial candidates based on semantic relevance between the query and the chunk's text snippet.
    * **Display:** Shows the final top N re-ranked results with enhanced metadata for readability.

## Setup

### Prerequisites

* Python 3.9+
* Access to a terminal or command prompt.
* Git installed.
* SSH key configured with your GitHub account (for cloning using the SSH URL). Alternatively, use the HTTPS URL from the GitHub repository page.
* API Keys for:
    * Mistral AI
    * Cohere
    * Pinecone

### Steps

1.  **Clone the Repository:**
    ```bash
    git clone git@github.com:PelegChen57522/SolidCAM.git
    cd SolidCAM
    ```
    *(Or use the HTTPS URL if preferred)*

2.  **Create and Activate Virtual Environment:**
    ```bash
    python -m venv venv
    # On Windows: venv\Scripts\activate
    # On macOS/Linux: source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create `.env` File:**
    Copy the `.env.example` file to `.env`:
    ```bash
    cp .env.example .env
    ```
    Edit the `.env` file and add your actual API keys:
    ```env
    MISTRAL_API_KEY=your_mistral_api_key_here
    COHERE_API_KEY=your_cohere_api_key_here
    PINECONE_API_KEY=your_pinecone_api_key_here
    # PINECONE_ENVIRONMENT=your_pinecone_environment # Optional
    ```
    **Important:** The `.env` file is listed in `.gitignore` and should **never** be committed to version control.

5.  **Set Up Pinecone Index:**
    * Log in to your Pinecone account.
    * Create a new index (or ensure an existing one is used).
    * **Index Name:** `solidcam-chatbot-image-embeddings` (This must match `PINECONE_INDEX_NAME` in the scripts)
    * **Dimension:** **`1024`** (to match Cohere `embed-v4.0`)
    * **Metric:** `cosine` (or your preferred metric)

6.  **Place Input PDF:**
    * Ensure the input PDF file (e.g., `Milling 2024 Machining Processes.pdf`) is placed in the correct location referenced by the `PDF_FILE_PATH` variable inside `MistralAI_OCR.py`. You might need to create a `pdf_files` directory.

## Usage

Run the scripts in the following order from your activated virtual environment:

1.  **Run OCR:**
    ```bash
    python MistralAI_OCR.py
    ```
    This generates `processed_solidcam_doc.json`.

2.  **Run Embedding and Storage:**
    ```bash
    python embed_and_store.py
    ```
    * **Important:** Check the `CLEAR_PINECONE_INDEX_BEFORE_RUN` flag in this script. Set it to `True` (default) to clear the index before adding new embeddings (recommended for the first run or after changing chunking/models). Set to `False` to add to an existing index (ensure dimensions match).

3.  **Run Test Queries:**
    ```bash
    python query_script.py
    ```
    This script executes test queries, performs re-ranking, and displays results with updated metadata format.

## Key Findings / Evaluation Summary

* The final workflow (Refined Header Chunking + Cohere Fused Embedding + Cohere Rerank) successfully embeds both text and images from the PDF.
* MistralAI OCR effectively extracts text, structure (headers), and Base64 images.
* Cohere `embed-v4.0`'s fused embedding capability successfully links text chunks with images referenced within them.
* The refined header chunking strategy (ignoring minor/list-like headers) improved retrieval relevance.
* Cohere Re-rank significantly improves the ranking precision, bringing the most relevant results (including those requiring combined text-image context) to the top for the test queries.
* The approach effectively maintains structural meaning for coherent retrieval, addressing the core requirement. Metadata includes the relevant header and 1-based page numbering for better readability.

## Limitations / Considerations

* **OCR Accuracy:** Retrieval quality depends on the OCR's accuracy in text/structure/image tag placement.
* **Chunking Impact:** The optimal chunking strategy can be document-dependent. Further tuning of the exclusion list or regex might be needed for different documents.
* **Reranker Dependency:** Best results rely on the re-ranking step (extra API call).
* **API Limits:** Trial keys may hit rate limits. Production keys are needed for sustained use.


