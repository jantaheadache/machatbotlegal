import streamlit as st
import google.generativeai as genai
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests # <--- ADDED THIS IMPORT

# --- Configuration and Setup ---

# URL to your publicly hosted search_index.pkl (YOUR URL IS HERE)
SEARCH_INDEX_URL = "https://huggingface.co/datasets/Jantaaaaa/jantaaaaamachatbotindex/resolve/main/search_index.pkl"
LOCAL_SEARCH_INDEX_PATH = "search_index.pkl" # This is where the app will save it temporarily

# Get Gemini API Key from environment or hardcoded value (Colab notebook injects it)
# It's better practice to use st.secrets or environment variables for deployed apps.
# For Colab, we rely on the notebook's ability to pass it or hardcode it for simplicity.
# !!! Ensure the API key is correctly set in the Colab notebook's Step 2 !!!
# The following section was modified to use st.secrets for deployment.
# It was previously: API_KEY = os.environ.get("GEMINI_API_KEY", "{gemini_api_key}")
if "GEMINI_API_KEY" in st.secrets:
    API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    st.error("Gemini API key not found in Streamlit secrets!")
    st.stop() # Stop the app if key is missing

# Configure Gemini with the retrieved API key
genai.configure(api_key=API_KEY)

st.set_page_config(
    page_title="M&A Agreement Analyzer",
    page_icon="⚖️",
    layout="centered"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        color: #1f4e79;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .disclaimer {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    .answer-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
    .stTextArea, .stTextInput {
        border: 1px solid #ced4da;
        border-radius: 0.25rem;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">M&A Agreement Analyzer</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
<strong>⚠️ Legal Disclaimer:</strong> This tool provides analysis based on document patterns and should not be considered legal advice. Always consult with qualified legal professionals for legal matters.
</div>
""", unsafe_allow_html=True)

# Initialize session state for the search index
if 'search_data' not in st.session_state:
    st.session_state.search_data = None
    st.session_state.setup_complete = False

# Setup (run once to load the TF-IDF index and configure Gemini)
# THIS ENTIRE BLOCK WAS MODIFIED TO DOWNLOAD THE INDEX FROM HUGGING FACE
if not st.session_state.setup_complete:
    with st.spinner("🔍 Downloading and loading M&A agreement database..."):
        try:
            # Download the index file
            if not os.path.exists(LOCAL_SEARCH_INDEX_PATH):
                st.write(f"Downloading data from {SEARCH_INDEX_URL}...")
                response = requests.get(SEARCH_INDEX_URL)
                response.raise_for_status() # Raise an error for bad status codes
                with open(LOCAL_SEARCH_INDEX_PATH, 'wb') as f:
                    f.write(response.content)
                st.write("Download complete.")
            else:
                st.write("Search index already downloaded from a previous run.")

            # Now load the index
            if os.path.exists(LOCAL_SEARCH_INDEX_PATH):
                with open(LOCAL_SEARCH_INDEX_PATH, 'rb') as f:
                    st.session_state.search_data = pickle.load(f)
                st.session_state.setup_complete = True
                st.success(f"✅ Database loaded successfully with {len(st.session_state.search_data['documents'])} chunks!")
            else:
                st.error("❌ Search index file not found after download attempt. Check the URL and permissions.")

        except Exception as e:
            st.error(f"❌ Error during setup: {e}")
            st.info("Please ensure your search_index.pkl is publicly accessible at the given URL.")

# --- Core Chatbot Logic (Using TF-IDF for retrieval) ---

def search_documents(query, top_k=5):
    if st.session_state.search_data is None:
        return []

    vectorizer = st.session_state.search_data['vectorizer']
    tfidf_matrix = st.session_state.search_data['tfidf_matrix']
    documents = st.session_state.search_data['documents']

    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]

    return [documents[i] for i in top_indices]

def ask_ma_question(question):
    try:
        relevant_docs = search_documents(question, top_k=5)

        if not relevant_docs:
            return "I couldn't find any relevant information in the provided M&A agreements for your question. Please try rephrasing or asking about a different topic."

        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""
        You are a NEW YORK MAGIC CIRCLE LAW FIRM M&A Lawyer. Analyze the following M&A agreement excerpts and answer the question. You may be asked legal questions, or to analyse clause or draft them.

        Provide a structured, professional response that:
        1. Directly answers the question.
        2. Cites specific clauses or terms when relevant.
        3. Notes any patterns or variations found.
        4. Mentions if information is limited or unclear based *only* on the provided context.
        5. Do NOT make up information. If the answer is not in the context, state that.

        Context from M&A agreements:
        {context}

        Question: {question}

        Professional Analysis:
        """

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error processing question: {e}"

# --- Streamlit UI Elements ---

st.subheader("Ask Your Question")
question = st.text_area(
    "Enter your M&A analysis question:",
    placeholder="Example: What are the typical indemnification periods in these agreements?",
    height=100
)

with st.expander("💡 Suggested Questions"):
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Deal Structure:**")
        st.write("• What deal structures are used?")
        st.write("• How is purchase price determined?")
        st.write("• What are the payment terms?")
    with col2:
        st.write("**Risk & Indemnity:**")
        st.write("• What are common representations?")
        st.write("• What indemnification terms exist?")
        st.write("• What are typical closing conditions?")

if st.button("Get Analysis"):
    if st.session_state.setup_complete:
        if question:
            with st.spinner("Analyzing agreements... This might take a moment."):
                answer = ask_ma_question(question)
                st.markdown(f'<div class="answer-box"><strong>Answer:</strong><br>{answer}</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter a question to get an analysis.")
    else:
        st.error("The database is still loading or failed to load. Please wait or check for errors above.")

st.markdown("---")
st.info("Built with Streamlit & Google Gemini AI | For legal professionals")
