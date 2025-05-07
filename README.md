# SolidCAM Documentation Chatbot - Multimodal RAG Implementation

This project, undertaken as part of the interview process at SolidCAM, aimed to develop and test an advanced workflow for the SolidCAM ChatBot. The primary objective was to enable the chatbot to understand and leverage both textual and visual information from PDF documentation using multimodal embeddings. This involved creating a Retrieval-Augmented Generation (RAG) system capable of processing the "Milling 2024 Machining Processes.pdf" document, preserving its structural meaning (including image-text associations), and providing accurate, context-aware responses through a user-friendly web interface.

The key requirements, as outlined by Ori Somekh, were to:

1.  Utilize Cohere's `embed-v4.0` model for multimodal embeddings.
2.  Store these embeddings in a Pinecone vector index.
3.  Preserve the structural integrity of the document, ensuring headers, subheaders, and their corresponding content (including images) remain associated.
4.  Build a RAG framework allowing users to query the documentation, with the chatbot maintaining conversational context.
5.  Assess the accuracy of the RAG system and the benefits of combined text and image vector context.

This project successfully addresses these requirements, culminating in a functional web-based chatbot.

**GitHub Repository:** [git@github.com:PelegChen57522/SolidCAM.git](git@github.com:PelegChen57522/SolidCAM.git)

## Final Workflow Overview

The implemented workflow consists of the following main stages:

1.  **PDF Processing (OCR & Structure Extraction):**

    - The `MistralAI_OCR.py` script processes the input PDF (`Milling 2024 Machining Processes.pdf`).
    - It uses the Mistral AI API (`mistral-ocr-latest` model) to extract structured text in Markdown format and associated image data (Base64 encoded) for each page.
    - The output is saved as a structured JSON file (`processed_solidcam_doc.json`).

2.  **Chunking, Embedding, and Storage:**

    - The `embed_and_store.py` script reads the `processed_solidcam_doc.json` file.
    - **Strategic Chunking:** It intelligently chunks the Markdown content based on H1, H2, and H3 headers. Specific non-semantic headers (e.g., "See Also", "Related Topics") and list-like headers are excluded to enhance semantic grouping and retrieval relevance.
    - **Image Association (Metadata):** Markdown image tags within each chunk are identified, and their filenames are stored in the metadata.
    - **Embedding with Cohere:** It uses the Cohere `embed-v4.0` model (via `cohere.ClientV2()`) to generate 1024-dimension embeddings for each text chunk. The `output_dimension` parameter is explicitly set to `1024`.
    - **Vector Storage (Pinecone):** The generated embeddings are stored in a Pinecone index (`solidcam-chatbot-image-embeddings`). Each vector's metadata includes:
      - `vector_id`: The unique ID of the chunk (e.g., `page1-chunk0`).
      - `page`: The 1-based page number in the PDF.
      - `header`: The main header associated with the chunk.
      - `text_snippet`: A snippet of the chunk's text content (used for display and by Langchain's `PineconeVectorStore`).
      - `image_ids`: A list of image filenames referenced in the chunk.
      - `has_images`: A boolean indicating if the chunk references images.

3.  **Retrieval-Augmented Generation (RAG) Chatbot with Web UI:**
    - The `app.py` script implements a Flask web application serving as the backend for the chatbot.
    - **Custom Embeddings for Queries:** A custom Langchain-compatible class (`CustomCohereEmbeddingsWithClientV2`) is used. This class leverages `cohere.ClientV2()` to ensure query embeddings are also generated at 1024 dimensions, matching the stored vectors.
    - **Langchain Orchestration:** It uses Langchain's `ConversationalRetrievalChain` to manage the RAG pipeline:
      - **Retrieval:** Fetches relevant document chunks from Pinecone.
      - **Reranking:** Employs `CohereRerank` (`rerank-english-v3.0`).
      - **Contextual Memory:** Utilizes `ConversationBufferMemory`.
      - **Generation:** Uses Cohere's `command-r` model (via `ChatCohere`).
    - **Web Interface:** The Flask app serves an `index.html` file (located in a `templates` folder) which provides a user-friendly chat interface built with HTML, Tailwind CSS, and JavaScript.

## Key Technologies Used

- **OCR:** Mistral AI (`mistral-ocr-latest`)
- **Embeddings:** Cohere (`embed-v4.0`)
- **Language Model (Generation):** Cohere (`command-r`)
- **Reranking:** Cohere (`rerank-english-v3.0`)
- **Vector Database:** Pinecone
- **Orchestration Framework:** Langchain
- **Web Backend:** Flask
- **Web Frontend:** HTML, Tailwind CSS, JavaScript

## Setup Instructions

### Prerequisites

- Python 3.9+
- Git (for cloning the repository)
- Access to a terminal or command prompt.
- API Keys for:
  - Mistral AI
  - Cohere
  - Pinecone

### Steps

1.  **Clone the Repository:**

    ```bash
    git clone git@github.com:PelegChen57522/SolidCAM.git
    cd SolidCAM
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
    Create a `requirements.txt` file (as provided separately) in the root of your project directory. Then, install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Create `.env` File:**
    Create a file named `.env` in the root of your project directory and add your API keys:

    ```env
    MISTRAL_API_KEY=your_mistral_ai_api_key
    COHERE_API_KEY=your_cohere_api_key
    PINECONE_API_KEY=your_pinecone_api_key
    ```

    **Important:** Ensure the `.env` file is listed in your `.gitignore`.

5.  **Set Up Pinecone Index:**

    - Log in to your Pinecone account.
    - Create a new index (or ensure an existing one is configured correctly).
    - **Index Name:** `solidcam-chatbot-image-embeddings`
    - **Dimension:** `1024`
    - **Metric:** `cosine`

6.  **Place Input PDF:**
    - Create a directory (e.g., `pdf_files`) in your project root.
    - Place the `Milling 2024 Machining Processes.pdf` file into this directory.
    - Update the `PDF_FILE_PATH` variable in `MistralAI_OCR.py` to point to this location (e.g., `pdf_files/"Milling 2024 Machining Processes.pdf"`).

## Running the Application

Execute the scripts in the following order from your activated virtual environment:

1.  **Step 1: Process PDF with MistralAI OCR**

    - This script converts the PDF to `processed_solidcam_doc.json`.

    ```bash
    python MistralAI_OCR.py
    ```

2.  **Step 2: Embed Content and Store in Pinecone**

    - **Important:** For the first run, or if you change the PDF/embedding logic, ensure `CLEAR_PINECONE_INDEX_BEFORE_RUN = True` in `embed_and_store.py`.

    ```bash
    python embed_and_store.py
    ```

3.  **Step 3: Run the Flask Web Server**

    ```bash
    python app.py
    ```

    - The server will typically run on `http://localhost:7001`. Check the terminal output for the exact URL.

4.  **Step 4: Access the Chatbot UI**
    - Open your web browser and navigate to the URL provided by the Flask server (e.g., `http://localhost:7001`).

## Key Achievements and Features

- **Multimodal Contextual Understanding:** The system processes PDFs with text and images. While not displaying images, embeddings capture context from text around images, aiding retrieval for image-related queries.
- **Preservation of Structural Meaning:** Chunking respects document structure (H1-H3 headers), associating content with relevant headings for coherent retrieval.
- **Effective RAG Implementation:**
  - **Accurate Retrieval:** Cohere `embed-v4.0` and Pinecone provide relevant document chunks.
  - **Enhanced Relevance with Reranking:** Cohere Rerank prioritizes the most relevant information.
  - **Conversational Context:** Langchain's `ConversationalRetrievalChain` maintains context across turns.
- **User-Friendly Web Interface:** A Flask and HTML/CSS/JS UI for interaction.
- **Consistent Embedding Dimensions:** Custom embedding logic ensures 1024 dimensions for both stored and query embeddings.
- **Detailed Source Attribution:** The UI displays retrieved source document information (page, header, and the vector ID).

## Addressing Interviewer's Evaluation Points

This project directly addresses the key areas Ori Somekh wished to assess:

- **Accuracy of RAG:** Demonstrated by the chatbot's ability to answer specific questions based on the PDF by retrieving and synthesizing information from correct document sections.
- **Benefits of Image and Text Vectors:** The system's ability to answer questions about content described alongside images (e.g., parameters from a diagram) shows the value of processing documents holistically.
- **Maintaining Context:** The chatbot handles follow-up questions effectively.
- **Preserving Structural Meaning:** Chunking and retrieval respect document headers.

This implementation provides a solid foundation for a powerful, context-aware chatbot for SolidCAM documentation.
