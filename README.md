# Sep10

# **README for Python Project: Chat with PDFs using Streamlit, Langchain, and OpenAI**

## **Project Overview**

This project presents an interactive Streamlit application titled "Chat with PDFs: Where Unstructured Data Meets Langchain 🦙". It allows users to upload PDF documents and ask questions, receiving answers based on the document's content. The app leverages Langchain, Unstructured, and OpenAI technologies for document processing and question answering.

## **Features**

1. **PDF Upload**: Users can upload PDF documents for analysis.
2. **Interactive Query Interface**: The app provides a chat-like interface where users can ask questions about the uploaded PDF.
3. **Chunk Size and Overlap Lines Adjustment**: Users can specify how the PDF is split into chunks and the overlap between these chunks for better context understanding.
4. **Top K Results**: Option to select the number of relevant chunks considered when generating an answer.
5. **Error Handling**: The app includes error handling for various issues like corrupt PDFs, indexing errors, and file not found errors.

## **Requirements**

- **`pdfplumber`**
- **`openai`**
- **`streamlit`**
- **`annoy`**
- **`langchain.document_loaders`**

## **Installation**

To install the required libraries, run:

```bash
pip install pdfplumber openai streamlit annoy langchain.document_loaders
```

## **Usage**

1. Run the Streamlit app using the command **`streamlit run [filename].py`**.
2. On the app interface:
    - Upload a PDF using the file uploader in the sidebar.
    - Adjust the chunk size and overlap lines for document processing.
    - Ask a question in the chat box about the uploaded document.
3. The app processes the PDF, creates embeddings, and uses OpenAI's models to answer the questions.

## **Functions Overview**

- **`handle_error`**: Manages various errors and displays appropriate messages to the user.
- **`setup_streamlit`**: Configures the Streamlit app's appearance and layout.
- **`setup_sidebar`**: Contains controls for PDF processing parameters and file uploading.
- **`init_session_state`**: Initializes session state for storing messages.
- **`render_chat_history`**: Renders the chat history on the Streamlit interface.
- **`process_user_input`**: Handles user input in the chat interface.
- **`setup_openai_api`**: Sets up the OpenAI API for embeddings and text generation.
- **`split_pdf`**: Splits the uploaded PDF into chunks for processing.
- **`extract_text_from_pdf`**: Extracts text from the PDF file and splits it into chunks.
- **`setup_annoy`**: Initializes the Annoy index for similarity search.
- **`create_openai_embedding`**: Creates embeddings for text chunks using OpenAI's models.
- **`upsert_to_annoy`**: Updates the Annoy index with new embeddings.
- **`query_annoy`**: Queries the Annoy index to find relevant chunks based on a question embedding.
- **`process_query_results`**: Processes the results returned by the Annoy query.
- **`generate_answer`**: Generates an answer based on the context data and user question.
- **`format_answer_in_markdown`**: Formats the plain text answer into Markdown.
- **`main`**: The main function where the app logic is executed.
