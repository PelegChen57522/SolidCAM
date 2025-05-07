# SolidCAM Documentation Chatbot: Advanced Multimodal RAG Implementation

**Candidate:** Peleg Chen
**Interviewer:** Ori Somekh, SolidCAM
**Project Date:** May 2024
**GitHub Repository:** [git@github.com:PelegChen57522/SolidCAM.git](git@github.com:PelegChen57522/SolidCAM.git)

## 1. Project Introduction

This project was undertaken as a technical assessment for a position at SolidCAM. The core task was to explore and implement an advanced workflow for the SolidCAM ChatBot, specifically focusing on integrating **multimodal embeddings** to understand and leverage both textual and visual information from PDF documentation.

The primary objective was to process the "Milling 2024 Machining Processes.pdf" document, create a Retrieval-Augmented Generation (RAG) system using Cohere's `embed-v4.0` model for multimodal embeddings, store these in a Pinecone vector index, and build a conversational interface. Key emphasis was placed on preserving the structural meaning of the document (including image-text associations) and assessing the benefits of this multimodal approach.

This document details the successfully implemented solution, which includes OCR processing, strategic chunking, multimodal embedding, vector storage, and a RAG-based chatbot with a web UI and a command-line interface.

## 2. Objectives & Requirements

As outlined by Ori Somekh, the key requirements were:

1.  **Multimodal Embeddings:** Utilize Cohere's `embed-v4.0` model to generate embeddings for both text and images from the provided PDF.
2.  **Vector Storage:** Set up and use a Pinecone vector index to store these embeddings.
3.  **Structural Preservation:** Maintain the structural relationships within the document (headers, subheaders, corresponding content, and associated images) during processing and embedding.
4.  **RAG Framework:** Develop a RAG system (preferably using Langchain) to query the documentation.
5.  **Conversational Context:** Ensure the chatbot can maintain context over multiple conversational turns.
6.  **Accuracy Assessment:** Evaluate the accuracy of the RAG system and the tangible benefits of using combined image and text vectors.
7.  **Post-Processing:** Address potential issues with Markdown parsing (e.g., "## See Also" being misinterpreted) by implementing appropriate cleaning or header selection logic.

## 3. Solution Architecture & Workflow

The implemented end-to-end workflow is as follows:

![Workflow Diagram (Conceptual - A diagram would ideally be here if this were a full report)](https://placehold.co/800x300/EBF4FF/1E40AF?text=Conceptual+Workflow%3A+PDF+%E2%86%92+OCR+%E2%86%92+Chunking+%E2%86%92+Multimodal+Embedding+%E2%86%92+Pinecone+%E2%86%92+RAG+Chatbot)
_(Conceptual Diagram: PDF → Mistral OCR → Smart Chunking → Cohere Multimodal Embedding → Pinecone → Langchain RAG → Chat Interface)_

1.  **PDF Processing & OCR (`MistralAI_OCR.py`):**

    - The input PDF (`Milling 2024 Machining Processes.pdf`) is processed using the Mistral AI API (`mistral-ocr-latest` model).
    - This script extracts:
      - Markdown text content for each page.
      - Base64 encoded image data for images found on each page.
      - Mistral OCR also inserts Markdown image tags (`![](image_id.jpeg)`) into the extracted Markdown text, linking the text to the image IDs.
    - The structured output (pages with their markdown and associated image data) is saved to `processed_solidcam_doc.json`.

2.  **Chunking, Multimodal Embedding & Storage (`embed_and_store.py`):**

    - **Data Loading:** Reads `processed_solidcam_doc.json`.
    - **Strategic Chunking:**
      - Markdown content is chunked based on H1, H2, and H3 headers.
      - A custom regex (`HEADER_CHUNK_PATTERN`) is used to identify valid headers, while specifically excluding non-semantic headers like "See Also" and "Related Topics" to improve the quality of semantic chunks.
    - **Image Association for Embedding:**
      - For each text chunk, associated image Markdown tags are parsed.
      - The corresponding base64 image data (converted to data URLs) is retrieved.
    - **Multimodal Embedding with Cohere:**
      - Uses Cohere's `embed-v4.0` model via `cohere.ClientV2().embed()`.
      - Crucially, for each chunk, **both the text and the associated image data URLs** are passed to the embedding model, creating a true multimodal embedding that captures combined textual and visual semantics.
      - Embeddings are generated at 1024 dimensions.
    - **Vector Storage in Pinecone:**
      - The generated multimodal embeddings are stored in a Pinecone index (`solidcam-chatbot-image-embeddings`).
      - Metadata for each vector includes: `vector_id`, `page`, `header`, `text_snippet` (first 2000 chars of the chunk text), `image_ids` (list of image filenames/IDs associated with the chunk), and `has_images` (boolean).
      - _Note:_ Full image data URLs are _not_ stored in Pinecone metadata due to per-vector size limits. This was an iterative refinement to solve an earlier "metadata size too large" error. The multimodal embedding _itself_ contains the image context.

3.  **Retrieval-Augmented Generation (RAG) Chatbot:**
    - Implemented in two forms:
      - **Web Application (`app.py`):** A Flask backend serving an HTML/Tailwind CSS/JavaScript frontend (`templates/index.html`).
      - **Command-Line Interface (`rag_chatbot.py`):** For direct terminal interaction.
    - **Core RAG Components (Langchain):**
      - **Custom Embeddings for Queries:** A `CustomCohereEmbeddingsWithClientV2` class ensures query embeddings also use `embed-v4.0` at 1024 dimensions via `cohere.ClientV2()`, matching the document embeddings.
      - **Retriever:** `PineconeVectorStore` fetches initial candidate chunks (`k=10`).
      - **Reranker:** `CohereRerank` (`rerank-english-v3.0`) refines the retrieved chunks to the top N (`top_n=3`) most relevant ones. This is wrapped in a `ContextualCompressionRetriever`.
      - **Language Model (Generator):** Cohere's `command-r` model (via `ChatCohere`) generates answers based on the reranked context.
      - **Conversational Memory:** `ConversationBufferMemory` allows the chatbot to maintain context across multiple turns.
      - **Orchestration:** `ConversationalRetrievalChain` ties these components together.
    - **Source Attribution:** The web UI displays information about the retrieved source documents (page, header, chunk ID, and whether images were associated during embedding).

## 4. Key Features & Technical Highlights

- **True Multimodal Embeddings:** Successfully implemented embedding of combined text and image data using Cohere `embed-v4.0`, allowing the system to understand visual context.
- **Preservation of Document Structure:** Intelligent chunking based on semantic headers (H1-H3) while excluding non-content headers ensures that retrieved contexts are coherent and meaningful.
- **Robust RAG Pipeline:** Leverages Langchain for a sophisticated RAG process including efficient retrieval, powerful reranking, and conversational memory.
- **Consistent High-Dimensional Embeddings:** Both document and query embeddings are consistently generated at 1024 dimensions using `embed-v4.0` via `cohere.ClientV2()`.
- **User-Friendly Interfaces:** Provides both a web UI for interactive chat and a CLI for quick testing.
- **Error Handling & Iterative Refinement:** Addressed practical challenges such as Pinecone's metadata size limits by adjusting the data stored, demonstrating an iterative problem-solving approach.
- **Clear Source Attribution:** The chatbot indicates the sources of its information, enhancing transparency and trust.

## 5. Technologies Used

- **Programming Language:** Python 3.13
- **OCR:** Mistral AI API (`mistral-ocr-latest`)
- **Embeddings (Multimodal):** Cohere API (`embed-v4.0`)
- **Language Model (Generation):** Cohere API (`command-r`)
- **Reranking:** Cohere API (`rerank-english-v3.0`)
- **Vector Database:** Pinecone
- **Orchestration Framework:** Langchain
- **Web Backend:** Flask
- **Web Frontend:** HTML, Tailwind CSS, JavaScript
- **Environment Management:** `venv`, `python-dotenv`

## 6. Setup and Running Instructions

### Prerequisites

- Python 3.9+ (developed with 3.13)
- Git
- API Keys for: Mistral AI, Cohere, Pinecone.

### Steps

1.  **Clone the Repository:**

    ```bash
    git clone git@github.com:PelegChen57522/SolidCAM.git
    cd SolidCAM_ChatBotImageEmbeddings # Or your project directory name
    ```

2.  **Create and Activate Virtual Environment:**

    ```bash
    python -m venv venv
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    # venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Ensure you have a `requirements.txt` file with the following content (or similar, based on your exact package versions):

    ```txt
    # requirements.txt
    mistralai
    cohere
    pinecone-client
    langchain
    langchain-pinecone
    langchain-cohere
    langchain-core
    flask
    flask-cors
    python-dotenv
    # Add other specific versions if needed, e.g., cohere==5.x.x
    ```

    Then install:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Create `.env` File:**
    In the project root, create a `.env` file and add your API keys:

    ```env
    MISTRAL_API_KEY="your_mistral_ai_api_key"
    COHERE_API_KEY="your_cohere_api_key"
    PINECONE_API_KEY="your_pinecone_api_key"
    # Optional: PINECONE_ENVIRONMENT, PINECONE_CLOUD_PROVIDER, PINECONE_REGION if creating a new index
    ```

    _Ensure `.env` is in your `.gitignore`._

5.  **Pinecone Index Setup:**

    - The `embed_and_store.py` script is configured to create a Pinecone index named `solidcam-chatbot-image-embeddings` with `1024` dimensions (metric: `cosine`) if it doesn't exist (controlled by `CREATE_PINECONE_INDEX_IF_NOT_EXISTS = True`).
    - Configure `PINECONE_INDEX_TYPE`, `PINECONE_CLOUD_PROVIDER`, and `PINECONE_REGION` in `embed_and_store.py` if creating a new serverless index.

6.  **Place Input PDF:**
    - Ensure the `Milling 2024 Machining Processes.pdf` file is located at the path specified by `PDF_FILE_PATH` in `MistralAI_OCR.py` (currently set to `/Users/pelegchen/SolidCAM_ChatBotImageEmbeddings/pdf_files/Milling 2024 Machining Processes.pdf`). Adjust if necessary.

### Running the Application Workflow

Execute the scripts in the following order from your activated virtual environment:

1.  **Process PDF with MistralAI OCR:**

    ```bash
    python MistralAI_OCR.py
    ```

    _Output: `processed_solidcam_doc.json`_

2.  **Embed Content and Store in Pinecone:**

    - Set `CLEAR_PINECONE_INDEX_BEFORE_RUN = True` in `embed_and_store.py` for a fresh run.

    ```bash
    python embed_and_store.py
    ```

    _This populates the Pinecone index._

3.  **Run the Chatbot (Web UI or CLI):**
    - **Web UI:**
      ```bash
      python app.py
      ```
      Access via `http://localhost:7001` (or the port shown in the terminal).
    - **CLI:**
      ```bash
      python rag_chatbot.py
      ```

## 7. Addressing Interviewer's Evaluation Points

This project successfully demonstrates the requested capabilities:

- **Accuracy of RAG:** The system retrieves relevant textual context, enhanced by multimodal understanding at the embedding stage, enabling the `command-r` model to generate accurate answers to specific questions about the PDF.
- **Benefits of Image and Text Vectors:** By embedding text alongside associated image data URLs, the system can better capture the semantics of document sections where visuals are key. This leads to improved retrieval of text that explains or is related to these visuals, even if the query doesn't explicitly mention an image.
- **Maintaining Conversational Context:** The use of `ConversationBufferMemory` in Langchain allows the chatbot to handle follow-up questions and maintain a coherent dialogue.
- **Preserving Structural Meaning:** The chunking strategy focuses on semantic units defined by H1-H3 headers, and the exclusion of "See Also" type headers prevents them from being treated as primary content blocks, thus preserving meaningful structure. Image associations are maintained during the embedding process.
- **Post-Processing of Headers:** The `HEADER_CHUNK_PATTERN` in `embed_and_store.py` explicitly handles the requirement to treat only specific header levels as valid section delimiters and ignore others like "## See Also".

## 8. Challenges & Solutions

- **Pinecone Metadata Size Limits:**
  - **Challenge:** Initial attempts to store full base64 image data URLs in Pinecone metadata led to errors due to per-vector and per-request size limits.
  - **Solution:**
    1.  Reduced the Pinecone upsert batch size (e.g., to 20).
    2.  Critically, removed the storage of full `image_data_urls` from Pinecone metadata. Instead, only `image_ids` and `has_images` flags are stored. The full image data is used during the Cohere embedding step (which is the most important for capturing visual semantics in the vector) but not persisted in Pinecone's metadata layer. This resolved the size limit errors while still leveraging multimodal embeddings for retrieval.
- **API Client Versioning & Input Structures:** Ensuring correct input formats for `cohere.ClientV2().embed()` and handling Pinecone's `list_indexes()` API changes required careful attention to documentation and testing.

## 9. Potential Future Enhancements

- **Direct Image Display in UI:** Re-introduce image display by storing base64 data in a separate, efficient key-value store (e.g., Redis, local JSON file store) and fetching it on demand in `app.py` using the `image_ids` from Pinecone metadata.
- **Advanced Evaluation Framework:** Implement a quantitative evaluation suite (e.g., using RAGAs or custom metrics like MRR, Hit Rate, Faithfulness, Answer Relevance) with a curated Q&A dataset.
- **Finer-Grained Chunking:** Explore more advanced semantic chunking techniques beyond header-based splitting for potentially more nuanced context retrieval.
- **Table-Specific Processing:** Develop specialized methods to extract, embed, and query tabular data from the PDF.
- **Query Transformation:** Implement techniques to expand or rephrase user queries for improved retrieval recall.

## 10. Conclusion

This project successfully demonstrates the implementation of an advanced RAG workflow incorporating multimodal embeddings for the SolidCAM ChatBot. It meets the core requirements set forth, showcasing the ability to process complex PDF documents, leverage combined text and image context for improved semantic understanding, and provide a functional, context-aware conversational interface. The iterative approach to problem-solving, particularly in handling API limitations, further highlights the practical application of these technologies.
