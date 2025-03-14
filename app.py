import streamlit as st
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Download necessary NLTK data
nltk.download('punkt')

# Initialize models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
generator = pipeline("text2text-generation", model="google/flan-t5-base")

# Streamlit UI
st.title("📖 AI-Powered PDF Q&A")

st.sidebar.header("Upload your PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    st.sidebar.success("PDF uploaded successfully!")

    def extract_text_from_pdf(pdf_file):
        """Extract text from an uploaded PDF file."""
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
        return full_text

    # Extract text
    pdf_text = extract_text_from_pdf(uploaded_file)

    # Chunk text
    def chunk_text(text, chunk_size=512):
        sentences = sent_tokenize(text)
        chunks, current_chunk = [], []

        for sentence in sentences:
            current_chunk.append(sentence)
            if len(" ".join(current_chunk)) > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    text_chunks = chunk_text(pdf_text)

    # Generate embeddings
    embeddings = embedding_model.encode(text_chunks)
    embeddings = np.array(embeddings).astype('float32')

    # Initialize FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    def search(query, top_k=2):
        """Retrieve top K relevant chunks using FAISS."""
        query_embedding = embedding_model.encode([query]).astype('float32')
        distances, indices = index.search(query_embedding, top_k)
        return [(text_chunks[i], distances[0][j]) for j, i in enumerate(indices[0])]

    # User Query
    user_query = st.text_input("🔍 Ask a question about the PDF:")

    if user_query:
        # Retrieve relevant chunks
        retrieved_results = search(user_query)
        retrieved_chunks = [result[0] for result in retrieved_results]

        # Generate AI response
        def generate_response(query, retrieved_chunks):
            context = "\n".join(retrieved_chunks)
            prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            result = generator(prompt, max_length=200)
            return result[0]['generated_text']

        response = generate_response(user_query, retrieved_chunks)

        st.subheader("📌 AI Answer:")
        st.write(response)
