import logging
import tempfile
import openai
import pdfplumber  # Import pdfplumber instead of PyPDF2
from annoy import AnnoyIndex
import streamlit as st
from langchain.document_loaders import UnstructuredAPIFileLoader
import sys

st.set_page_config(
    page_title="Chat with legal docs, powered by Langchain and Unstructured. Sep 10",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
)

print("Running on Python version:", sys.version)

# Initialize logging
logging.basicConfig(level=logging.INFO)

import pdfplumber
from pdfminer.pdfparser import PDFSyntaxError

def handle_error(e):
    if isinstance(e, PDFSyntaxError):
        st.error("The uploaded PDF appears to be corrupt. Please upload a valid file.")
    elif isinstance(e, KeyError):
        st.error("There was an issue with indexing. Please try again.")
    elif isinstance(e, FileNotFoundError):
        st.error("The specified file was not found. Please check the file path.")
    else:
        st.error(f"An unexpected error occurred: {str(e)}")

# Streamlit setup
def setup_streamlit():
    st.title("Chat with Legal PDFs: Where Unstructured Data Meets Langchain 🦙")
    st.info("""
    📃 Welcome to this interactive Streamlit app! Here, you can query uploaded PDFs in real-time. 
    We leverage cutting-edge NLP technologies like [Langchain](https://docs.langchain.com/docs/), [Unstructured](https://unstructured.io/), and [OpenAI](https://platform.openai.com/docs/) to sift through your documents and fetch the information you need.
    """)

    # Interactive Tutorial
    with st.expander("📚 How to Use This App", expanded=False):
        st.markdown("""
        Follow these simple steps to get started:

        1. **Upload a PDF**: Locate the file uploader in the sidebar and select the PDF document you wish to query.
        2. **Set Chunk Size**: Adjust the slider in the sidebar to set the number of pages processed together.
        3. **Set Overlap Lines**: Use another slider to define the number of overlapping lines between chunks, ensuring context is preserved.
        4. **Ask a Question**: Type your query into the chat box below to get answers based on the uploaded document.
        """)


# Sidebar setup
def setup_sidebar():
    with st.sidebar:
        st.subheader("Controls")
        
        # Chunk Size Slider
        chunk_size = st.slider("📄 Select Chunk Size", min_value=1, max_value=4, value=2)
        with st.expander("About Chunk Size"):  # Dropdown for Chunk Size
            st.write("""
            The **Chunk Size** parameter specifies the number of consecutive pages from the PDF that will be grouped together into a single text chunk for processing. 
            A larger chunk size may improve the context for your queries but may also increase processing time.
            """)
        
        # Overlap Lines Slider
        overlap_lines = st.slider("🔗 Select Overlap Lines", min_value=0, max_value=5, value=2)
        with st.expander("About Overlap Lines"):  # Dropdown for Overlap Lines
            st.write("""
            The **Overlap Lines** parameter allows you to set how many lines of text at the end of one chunk will overlap with the beginning of the next chunk. 
            This ensures that sentences or information split between two chunks are not lost.
            """)
        
        # Top K Results Slider
        top_k_results = st.slider("🎯 Select Top K Results for Query", min_value=1, max_value=5, value=3)
        with st.expander("About Top K Results"):  # Dropdown for Top K Results
            st.write("""
            The **Top K Results** parameter specifies the number of most relevant chunks that you want to be considered when generating an answer to your query.
            """)
        
        with st.form("file_upload_form", clear_on_submit=True):
            file = st.file_uploader("📤 Upload your PDF Document", type=['pdf'])
            submit_button = st.form_submit_button("Submit")
    
    return file, submit_button, chunk_size, overlap_lines, top_k_results


# Initialize session state
def init_session_state():
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Ask me a question about the documents!"}]

# Render chat history
def render_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(message["content"], unsafe_allow_html=True)
            else:
                st.write(message["content"])

# Process user input
def process_user_input():
    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})

# OpenAI API setup
def setup_openai_api():
    openai.api_key = st.secrets.openai_key

@st.cache_data
def split_pdf(input_pdf, chunk_size=3, overlap_lines=5):
    with pdfplumber.open(input_pdf) as pdf:  # Use pdfplumber to open the PDF
        num_pages = len(pdf.pages)
        chunks = []
        page_ranges = []  # Store page ranges
        overlapping_text = ""

        for start in range(0, num_pages, chunk_size):
            text_chunk = overlapping_text
            end = min(start + chunk_size, num_pages)  # Calculate the end page for this chunk
            for i in range(start, end):
                page = pdf.pages[i]
                text_chunk += page.extract_text()  # Use pdfplumber's extract_text method

            # Update overlapping_text for the next iteration
            lines = text_chunk.split('\n')
            overlapping_text = '\n'.join(lines[-overlap_lines:]) if len(lines) > overlap_lines else ""

            chunks.append(text_chunk)
            page_ranges.append(f"{start+1}-{end}")  # 1-based page numbering

    return {"chunks": chunks, "page_ranges": page_ranges}  # Return as a dictionary


@st.cache_data
def extract_text_from_pdf(file, chunk_size=2, overlap_lines=2):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tfile:
            tfile.write(file.read())
            temp_file_path = tfile.name

        with open(temp_file_path, "rb") as f:
            result_dict = split_pdf(f, chunk_size, overlap_lines)  # Get the dictionary
            chunks = result_dict["chunks"]
            page_ranges = result_dict["page_ranges"]

        return {"chunks": chunks, "page_ranges": page_ranges}  # Return a dictionary
    except Exception as e:
        logging.exception("Exception occurred")
        handle_error(f"Error opening PDF file: {e}")
        return {}

# Function to set up Annoy index
def setup_annoy(dimension=1536):
    index = AnnoyIndex(dimension, 'angular')
    return index

# Function to create OpenAI embedding
def create_openai_embedding(text, max_words=300):
    if not text:
        handle_error("Input text is empty or None.")
        return None

    try:
        words = text.split()
        chunks = [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
        embeddings = []
        for chunk in chunks:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=chunk
            )
            embeddings.append(response["data"][0]["embedding"])
        embedding = [sum(x) / len(x) for x in zip(*embeddings)]
        return embedding
    except Exception as e:
        handle_error(f"Error creating OpenAI embedding: {e}")
        return None

# Function to upsert to Annoy index
def upsert_to_annoy(index, i, embedding, text):
    index.add_item(i, embedding)
    st.session_state.text_storage[i] = text


# Function to query Annoy index
def query_annoy(index, question_embedding, top_k=3):  # Added top_k parameter
    try:
        query_results = index.get_nns_by_vector(question_embedding, top_k, include_distances=True)  # Use top_k
        return query_results
    except Exception as e:
        handle_error(f"Error querying Annoy: {e}")
        return [], []

# Function to process query results
def process_query_results(query_results):
    results, scores = query_results
    context_data = []
    for i, result in enumerate(results):
        context_data.append(f"{result}: {scores[i]} - {st.session_state.text_storage[result]}")
    return context_data

# Function to generate an answer
def generate_answer(context_data, user_question):
    prompt = f"Context: {', '.join(context_data)}\nQuestion: {user_question}\nAnswer:"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a detail-oriented, friendly research assistant. Your task is to synthesize the provided information (which consists of OCR'ed portions of content from the user's uploaded PDF document) to answer the user's questions accurately, concisely, and clearly. Please use Markdown formatting like **bold**, *italics*, and `code` to enhance readability. Your answers should be organized, responsive, and accurate."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=3600,
            n=1,
            stop=None,
            temperature=0.0,
        )
        # Extract the plain text answer
        plain_answer = response.choices[0].message['content'].strip()

        # Call the new function to format the plain text answer in Markdown
        markdown_answer = format_answer_in_markdown(plain_answer, user_question)


        return markdown_answer  # Return the Markdown-formatted answer
    except Exception as e:
        handle_error(f"Error generating answer: {e}")
        return None

def format_answer_in_markdown(plain_answer, user_question):
    try:
        # Include the user's original question for context
        prompt = f"Original Question: {user_question}\nAnswer: {plain_answer}\nPlease format the answer in Markdown."

        # Make an API call to GPT-3.5 Turbo for Markdown formatting
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                    "role": "system",
                    "content": "Your role is to enhance the formatting, spelling/grammar, and organization of the provided text to optimize its readability, structure, and clarity. The text provided is an answer generated by an AI assistant to a user query based on the user's uploaded PDF document. Make effective use of Markdown syntax like **bold**, *italics*, and `code` to make the answer easy to read and understand."

                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=4000,
            n=1,
            stop=None,
            temperature=0.0,
        )

        # Extract the Markdown-formatted answer
        markdown_answer = response.choices[0].message['content'].strip()

        return markdown_answer

    except Exception as e:
        handle_error(f"Error in formatting answer in Markdown: {e}")
        return None

# Main function
def main():
    # Setup code here
    setup_streamlit()
    init_session_state()
    setup_openai_api()


    # Query PDF
    file, submit_button, chunk_size, overlap_lines, top_k_results = setup_sidebar() # Added top_k_results
    render_chat_history()
    process_user_input()
        

    # Initialize 'annoy_index' and 'index_built' if they're not already in session state
    if 'annoy_index' not in st.session_state:
        st.session_state.annoy_index = setup_annoy()
        st.session_state.index_built = False

    # Initialize 'text_storage' if it's not already in session state
    if 'text_storage' not in st.session_state:
        st.session_state.text_storage = {}

    if file is not None:
        with st.spinner('Extracting text from PDF...'):
            extracted_data = extract_text_from_pdf(file, chunk_size, overlap_lines)
            st.session_state.text_chunks = extracted_data["chunks"]
            st.session_state.page_ranges = extracted_data["page_ranges"]  # Save the page ranges if needed

        with st.spinner('Creating embeddings and updating index...'):
            progress_bar = st.progress(0)
            for i, text_chunk in enumerate(st.session_state.text_chunks):
                if text_chunk:
                    embedding = create_openai_embedding(text_chunk)
                    if embedding is not None:
                        upsert_to_annoy(st.session_state.annoy_index, i, embedding, text_chunk)
                progress_bar.progress((i + 1) / len(st.session_state.text_chunks))

        # Build Annoy index if not already built
        if not st.session_state.index_built:
            st.session_state.annoy_index.build(50)
            st.session_state.index_built = True

        st.success("Setup complete!")

    # New code to process user's question and generate answer
    if 'messages' in st.session_state and st.session_state.messages[-1]["role"] == "user":
        user_question = st.session_state.messages[-1]["content"]
        question_embedding = create_openai_embedding(user_question)
        if question_embedding is not None:
            query_results = query_annoy(st.session_state.annoy_index, question_embedding, top_k=top_k_results)
            context_data = process_query_results(query_results)


            # Generate the plain text answer
            plain_answer = generate_answer(context_data, user_question)
            
            # Format the plain text answer in Markdown
            markdown_answer = format_answer_in_markdown(plain_answer, user_question)

            if markdown_answer:
                st.session_state.messages.append({"role": "assistant", "content": markdown_answer})
                render_chat_history()

if __name__ == '__main__':
    main()
