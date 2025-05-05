# SolidCAM Chatbot - Multimodal Embedding Workflow Test

## Project Goal

This project implements and tests a new workflow for generating multimodal embeddings (text and images) from PDF documents for the SolidCAM ChatBot. The goal was to leverage newer AI models (MistralAI OCR and Cohere Embed v4) capable of understanding both text and visual elements, addressing the limitation of previous workflows that excluded images.

This task was assigned by Ori Somekh (SolidCAM) to evaluate this approach using the provided `Milling 2024 Machining Processes.pdf` document. A key requirement was to **preserve the structural meaning** of the document, ensuring headers, subheaders, and their corresponding content (including images) remain associated for coherent retrieval. This was successfully achieved through a combination of filtered header chunking, Cohere's fused embedding, and Cohere's reranking capabilities.

## Final Workflow Overview

The implemented workflow consists of the following main steps:

1.  **PDF Processing (OCR):** The `MistralAI_OCR.py` script uploads the input PDF (`Milling 2024 Machining Processes.pdf`) to the Mistral AI API. It uses the `mistral-ocr-latest` model to extract structured text (Markdown) and associated image data (Base64 encoded) for each page. The output is saved as a structured JSON file (`processed_solidcam_doc.json`). This step requires a Mistral AI API key.
2.  **Chunking & Fused Embedding:** The `embed_and_store.py` script reads the `processed_solidcam_doc.json` file.
    - **Chunking:** It chunks the Markdown content based on H1, H2, and H3 headers, **excluding** specific non-content headers (like "See Also", "Related Topics") to improve semantic grouping (Filtered Header Chunking).
    - **Image Association:** It identifies Markdown image tags within each chunk and associates the corresponding Base64 image data (formatted as Data URLs).
    - **Fused Embedding:** It uses the Cohere `embed-v4.0` model (via `ClientV2`) with the `inputs` parameter to create a single embedding vector for each chunk, combining the chunk's text and any associated image data URLs. This captures the text-image relationship.
    - **Storage:** The generated 1024-dimension vectors and associated metadata (page, text snippet, image IDs) are uploaded to a specified Pinecone index. This step requires Cohere and Pinecone API keys.
3.  **Querying & Re-ranking:** The `query_script.py` script provides a way to test retrieval:
    - **Query Embedding:** Embeds a text query using Cohere `embed-v4.0` (`input_type="search_query"`).
    - **Initial Retrieval:** Fetches an initial set of candidate chunks (e.g., top 10) from Pinecone based on vector similarity.
    - **Re-ranking:** Uses Cohere's `rerank-english-v3.0` model to re-order the initial candidates based on semantic relevance between the query and the chunk's text snippet.
    - **Display:** Shows the final top N re-ranked results with metadata.

## Setup

### Prerequisites

- Python 3.9+
- Access to a terminal or command prompt.
- API Keys for:
  - Mistral AI
  - Cohere
  - Pinecone

### Steps

1.  **Clone the Repository (If Applicable):**

    ```bash
    # git clone <repository-url>
    # cd <repository-directory>
    ```

2.  **Create and Activate Virtual Environment:**

    ```bash
    python -m venv venv
    # On Windows: venv\Scripts\activate
    # On macOS/Linux: source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Create a `requirements.txt` file with the following content:

    ```txt
    mistralai
    cohere
    pinecone-client
    python-dotenv
    requests # Usually a dependency of other libs, but good to include
    numpy # Often used implicitly or helpful for analysis
    ```

    Then install:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Create `.env` File:**
    Create a file named `.env` in the project's root directory and add your API keys:

    ```env
    MISTRAL_API_KEY=your_mistral_api_key_here
    COHERE_API_KEY=your_cohere_api_key_here
    PINECONE_API_KEY=your_pinecone_api_key_here
    # PINECONE_ENVIRONMENT=your_pinecone_environment # e.g., aws-us-east-1 (Often not needed with latest client)
    ```

5.  **Set Up Pinecone Index:**

    - Log in to your Pinecone account.
    - Create a new index (or ensure an existing one is used).
    - **Index Name:** `solidcam-chatbot-image-embeddings` (or update `PINECONE_INDEX_NAME` in the scripts)
    - **Dimension:** **`1024`** (to match Cohere `embed-v4.0`)
    - **Metric:** `cosine` (or your preferred metric)

6.  **Configure PDF Path:**
    - Verify the `PDF_FILE_PATH` variable in `MistralAI_OCR.py` points to the correct location of your input PDF.

## Usage

Run the scripts in the following order from your activated virtual environment:

1.  **Run OCR:**

    ```bash
    python MistralAI_OCR.py
    ```

    This will generate the `processed_solidcam_doc.json` file containing the OCR output with Base64 images.

2.  **Run Embedding and Storage:**

    ```bash
    python embed_and_store.py
    ```

    - **Important:** Check the `CLEAR_PINECONE_INDEX_BEFORE_RUN` flag in this script. Set it to `True` if you want to clear the index before adding new embeddings (recommended for the first run or after changing chunking/models). Set to `False` to add to an existing index (ensure dimensions match).
    - This script will chunk the JSON data, embed text/images using Cohere, and upload vectors to Pinecone.

3.  **Run Test Queries:**
    ```bash
    python query_script.py
    ```
    This script will execute the predefined test queries against the populated Pinecone index, using Cohere for query embedding and re-ranking, and display the results.

## Key Findings / Evaluation Summary

- The final workflow (Filtered Header Chunking + Cohere Fused Embedding + Cohere Rerank) successfully embeds both text and images from the PDF.
- MistralAI OCR effectively extracts text, structure (headers), and Base64 images.
- Cohere `embed-v4.0`'s fused embedding capability successfully links text chunks with images referenced within them.
- The "Filtered Header Chunking" strategy (ignoring minor headers like "See Also") improved retrieval relevance compared to basic header chunking.
- Cohere Re-rank significantly improves the ranking precision, bringing the most relevant results (including those requiring combined text-image context) to the top for most test queries.
- The approach effectively maintains structural meaning for coherent retrieval, addressing the core requirement.

## Limitations / Considerations

- **OCR Accuracy:** The quality of the final embeddings and retrieval depends on the accuracy of the initial OCR step in correctly extracting text, identifying structure (headers), and placing image references accurately in the Markdown flow.
- **Chunking Impact:** While the filtered header approach worked well here, the optimal chunking strategy can be document-dependent.
- **Reranker Dependency:** The best results rely on the re-ranking step, which adds an extra API call (and associated latency/cost) per query.
- **API Limits:** Using Trial API keys may result in rate limits, as observed during testing. Production keys are needed for sustained use.
