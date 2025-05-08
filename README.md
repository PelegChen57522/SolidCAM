# SolidCAM Documentation Chatbot: Advanced Multimodal RAG Implementation

**Candidate:** Peleg Chen
**Interviewer:** Ori Somekh, SolidCAM
**Project Date:** May 2025
**GitHub Repository:** [git@github.com:PelegChen57522/SolidCAM.git](git@github.com:PelegChen57522/SolidCAM.git)

## 1. Project Introduction

This project was undertaken as a technical assessment for a position at SolidCAM. The core task was to explore and implement an advanced workflow for the SolidCAM ChatBot, specifically focusing on integrating **multimodal embeddings** to understand and leverage both textual and visual information from PDF documentation.

The primary objective was to process the "Milling 2024 Machining Processes.pdf" document, create a Retrieval-Augmented Generation (RAG) system using Cohere's `embed-v4.0` model for multimodal embeddings, store these in a Pinecone vector index, and build a conversational interface. Key emphasis was placed on preserving the structural meaning of the document (including image-text associations) and assessing the benefits of this multimodal approach.

Going beyond the initial requirements, this project successfully implemented and tested **multimodal generation** using Cohere's `Aya Vision` model (`c4ai-aya-vision-32b`), allowing the chatbot to directly analyze images provided as context when formulating answers. This document details the final solution, including OCR, strategic chunking, multimodal embedding and generation, vector storage, and a RAG-based chatbot with web UI and CLI interfaces.

## 2. Objectives & Requirements

As outlined by Ori Somekh, the key requirements were:

1.  **Multimodal Embeddings:** Utilize Cohere's `embed-v4.0` model to generate embeddings for both text and images from the provided PDF. (Achieved)
2.  **Vector Storage:** Set up and use a Pinecone vector index to store these embeddings. (Achieved)
3.  **Structural Preservation:** Maintain the structural relationships within the document (headers, subheaders, corresponding content, and associated images) during processing and embedding. (Achieved via header-based chunking and image association logic)
4.  **RAG Framework:** Develop a RAG system (preferably using Langchain) to query the documentation. (Achieved)
5.  **Conversational Context:** Ensure the chatbot can maintain context over multiple conversational turns. (Achieved via Langchain Memory)
6.  **Accuracy Assessment:** Evaluate the accuracy of the RAG system and the tangible benefits of using combined image and text vectors. (Achieved through testing, including specific vision model tests)
7.  **Post-Processing:** Address potential issues with Markdown parsing (e.g., "## See Also" being misinterpreted) by implementing appropriate cleaning or header selection logic. (Achieved via regex in chunking)

**Extended Goal:** Implement and test multimodal _generation_ using a vision-capable model (`Aya Vision`) to directly analyze image context. (Achieved using `c4ai-aya-vision-32b`)

## 3. Solution Architecture & Workflow

The implemented end-to-end workflow is as follows:

![Workflow Diagram (Conceptual - A diagram would ideally be here if this were a full report)](https://placehold.co/800x300/EBF4FF/1E40AF?text=Conceptual+Workflow%3A+PDF+%E2%86%92+OCR+%E2%86%92+Chunking+%E2%86%92+Multimodal+Embedding+%E2%86%92+Pinecone+%E2%86%92+RAG+Chatbot+with+Vision+Generation)
_(Conceptual Diagram: PDF → Mistral OCR → Smart Chunking → Cohere Multimodal Embedding → Pinecone → Langchain RAG (Retrieval+Rerank) → Aya Vision Generation → Chat Interface)_

1.  **PDF Processing & OCR (`MistralAI_OCR.py`):**

    - Processes the input PDF using Mistral AI API (`mistral-ocr-latest`).
    - Extracts Markdown text and Base64 image data per page.
    - Mistral OCR inserts Markdown image tags (`![](image_id.jpeg)`) linking text to image IDs.
    - Output saved to `processed_solidcam_doc.json`.

2.  **Chunking, Multimodal Embedding & Storage (`embed_and_store.py`):**

    - Reads `processed_solidcam_doc.json`.
    - **Strategic Chunking:** Chunks Markdown based on H1-H3 headers, excluding non-semantic headers (e.g., "See Also") using regex.
    - **Image Association:** Parses image tags within chunks to identify associated images.
    - **Multimodal Embedding:** Uses Cohere `embed-v4.0` via `cohere.ClientV2().embed()`, passing **both text and associated image data URLs** to create 1024-dimension multimodal embeddings.
    - **Vector Storage:** Stores embeddings in Pinecone (`solidcam-chatbot-image-embeddings`). Metadata includes `vector_id`, `page`, `header`, `text_snippet` (2000 chars), `image_ids`, and `has_images`. Full image data URLs are _omitted_ from metadata to respect Pinecone limits.

3.  **Retrieval-Augmented Generation (RAG) Chatbot with Vision Generation:**
    - Implemented via Flask Web App (`app.py` - Version `app_py_v8_vision_prompt_tuning`) and CLI (`rag_chatbot.py`).
    - **Core RAG Components (Langchain & Cohere):**
      - **Query Embedding:** Custom Langchain class using `embed-v4.0` at 1024 dimensions.
      - **Retriever:** `PineconeVectorStore` fetches initial candidates (`k=10`).
      - **Reranker:** `CohereRerank` (`rerank-english-v3.0`) refines to top N (`top_n=3`).
      - **Context Formatting:** A custom function (`format_context_for_aya_vision`) prepares the final context for the generation model. It includes text snippets from top-N documents (with enhanced labeling) and **image data URLs (retrieved from the loaded JSON) only from the top-ranked document**, respecting the **4-image API limit** for Aya Vision.
      - **Language Model (Generator):** Cohere's **`c4ai-aya-vision-32b`** model (via `ChatCohere`) generates answers, receiving the formatted multimodal context (text + prioritized images). Includes a refined system prompt guiding the model to synthesize text and visual information.
      - **Fallback Mechanism:** If the Langchain `ChatCohere` wrapper fails with the multimodal input, the code attempts a fallback using the native `cohere.ClientV2().chat()` SDK call.
      - **Conversational Memory:** `ConversationBufferMemory` maintains dialogue history.
    - **Source Attribution & UI:** The web UI displays the bot's answer and source information (page, header, ID, image presence). Image thumbnails are displayed by retrieving data URLs on-the-fly in `app.py`.

## 4. Key Features & Technical Highlights

- **End-to-End Multimodal RAG:** Successfully built a pipeline that leverages multimodality in _both_ embedding/retrieval (`embed-v4.0`) and generation (`Aya Vision`).
- **Direct Image Analysis for Generation:** The chatbot's generation step directly processes image content alongside text, enabling answers based on visual information.
- **Intelligent Context Management:** Prioritizes visual context from the most relevant retrieved document and strictly adheres to the Cohere API's 4-image limit for vision models. Enhanced context labeling and system prompts aim to improve model focus.
- **Robust Implementation:** Includes fallback logic for LLM calls and addresses practical API limitations (Pinecone metadata size, Cohere image limits).
- **Preservation of Document Structure:** Chunking respects semantic document structure.
- **Consistent High-Dimensional Embeddings:** Ensures 1024 dimensions for both document and query embeddings.
- **User-Friendly Interfaces:** Provides both Web UI and CLI.
- **Clear Source Attribution:** Enhances transparency.

## 5. Technologies Used

- **Programming Language:** Python 3.13
- **OCR:** Mistral AI API (`mistral-ocr-latest`)
- **Embeddings (Multimodal):** Cohere API (`embed-v4.0`)
- **Language Model (Vision Generation):** Cohere API (`c4ai-aya-vision-32b`)
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
    Ensure `requirements.txt` exists (see previous versions for content) and run:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Create `.env` File:**
    In the project root, create `.env` with your API keys:

    ```env
    MISTRAL_API_KEY="your_mistral_ai_api_key"
    COHERE_API_KEY="your_cohere_api_key"
    PINECONE_API_KEY="your_pinecone_api_key"
    ```

    _Ensure `.env` is in your `.gitignore`._

5.  **Pinecone Index Setup:**

    - The `embed_and_store.py` script can create the index (`solidcam-chatbot-image-embeddings`, 1024 dims, cosine) if `CREATE_PINECONE_INDEX_IF_NOT_EXISTS = True`. Configure type/cloud/region as needed.

6.  **Place Input PDF:**
    - Ensure `Milling 2024 Machining Processes.pdf` is at the path specified in `MistralAI_OCR.py`.

### Running the Application Workflow

Execute in order from your activated virtual environment:

1.  **Process PDF:**

    ```bash
    python MistralAI_OCR.py
    ```

    _Output: `processed_solidcam_doc.json`_

2.  **Embed & Store:**

    - Set `CLEAR_PINECONE_INDEX_BEFORE_RUN = True` in `embed_and_store.py` for a fresh run.

    ```bash
    python embed_and_store.py
    ```

    _Populates Pinecone._

3.  **Run Chatbot:**
    - **Web UI:**
      ```bash
      python app.py
      ```
      Access via `http://localhost:7001`.
    - **CLI:**
      ```bash
      python rag_chatbot.py
      ```

## 7. Addressing Interviewer's Evaluation Points & Testing Insights

This project successfully addresses the evaluation points:

- **Accuracy of RAG:** The system retrieves relevant context, enhanced by multimodal embeddings. The generation step, using the powerful `c4ai-aya-vision-32b` model, attempts to synthesize text and visual information from the highest-ranked source.
- **Benefits of Image and Text Vectors:** Multimodal embeddings (`embed-v4.0`) demonstrably improve retrieval relevance. The Aya Vision generation step directly leverages retrieved images (prioritized from the top source), showcasing the potential for deeper visual understanding.
- **Maintaining Conversational Context:** Handled effectively via Langchain memory.
- **Preserving Structural Meaning:** Achieved through header-based chunking and careful exclusion of non-semantic headers.
- **Post-Processing of Headers:** Handled via regex during chunking.

## 8. Challenges & Solutions

- **Pinecone Metadata Size Limits:** Solved by removing base64 data URLs from metadata and storing only essential identifiers (`image_ids`, `has_images`). Image data is retrieved from the source JSON file on demand for the generation step.
- **Cohere Aya Vision Image Limit:** Implemented logic in `app.py` to prioritize images from the top-ranked document and strictly limit the number of images sent per API call to 4.
- **API Client/Langchain Integration:** Ensured correct input formats for different Cohere models (`embed-v4.0` vs. `Aya Vision`) and added native SDK fallback for robustness (though not needed in final testing).
- **Vision Model Accuracy:** Iterated on the generation model (8b vs 32b) and prompting strategy to improve visual interpretation, acknowledging current limitations.

## 10. Conclusion

This project successfully delivers an advanced, end-to-end multimodal RAG system that not only meets the original requirements but also explores the cutting-edge capabilities of Cohere's Aya Vision model for generation. It demonstrates a strong understanding of multimodal AI concepts, practical implementation skills in integrating various APIs and frameworks (Mistral, Cohere, Pinecone, Langchain), and the ability to iteratively solve real-world technical challenges like API limitations and model behaviour nuances. The resulting chatbot showcases the potential and provides a realistic assessment of leveraging both text and images for enhanced document understanding within the SolidCAM context, offering valuable insights for future development.
