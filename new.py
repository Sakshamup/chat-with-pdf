import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
from gtts import gTTS
import tempfile
import re

# Load environment variables
load_dotenv()

# Language mapping for gTTS (text-to-speech)
LANGUAGE_MAPPING = {
    "English": "en",
    "Hindi": "hi", 
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Arabic": "ar",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Portuguese": "pt",
    "Russian": "ru",
    "Italian": "it",
}

# Configure Google API
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🚨 GOOGLE_API_KEY not found! Please set your API key in environment variables.")
        st.stop()
    
    genai.configure(api_key=api_key)
    st.success("✅ Google API configured successfully!")
except Exception as e:
    st.error(f"🚨 API Configuration Error: {str(e)}")
    st.stop()

# Page config
st.set_page_config(
    page_title="AI PDF Chatbot", 
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, Modern CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .main .block-container {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        max-width: 1200px;
    }
    
    /* Header */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.125rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Upload Section */
    .upload-container {
        background: #f9fafb;
        border: 2px dashed #d1d5db;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .upload-container h3 {
        color: #374151;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .upload-container p {
        color: #6b7280;
        margin-bottom: 1rem;
    }
    
    /* Chat Messages */
    .user-message {
        background: #3b82f6;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 6px 18px;
        margin: 1rem 0;
        margin-left: 2rem;
        font-weight: 500;
    }
    
    .bot-message {
        background: #f3f4f6;
        color: #1f2937;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 18px 6px;
        margin: 1rem 0;
        margin-right: 2rem;
        border-left: 4px solid #3b82f6;
    }
    
    /* Buttons */
    .stButton > button {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: #2563eb;
        transform: translateY(-1px);
    }
    
    /* Input */
    .stTextInput > div > div > input {
        border: 2px solid #d1d5db;
        border-radius: 6px;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        outline: none;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
    }
    
    section[data-testid="stSidebar"] h3 {
        color: #1f2937;
        font-weight: 600;
        padding: 0.5rem 0;
        border-bottom: 2px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    
    /* Feature boxes */
    .feature-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.2s;
        margin-bottom: 1rem;
    }
    
    .feature-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
        transform: translateY(-2px);
    }
    
    /* File uploader */
    div[data-testid="stFileUploader"] {
        background: #f9fafb;
        border: 2px dashed #d1d5db;
        border-radius: 8px;
        padding: 2rem;
    }
    
    div[data-testid="stFileUploader"]:hover {
        border-color: #3b82f6;
        background: #eff6ff;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        border: 2px solid #d1d5db;
        border-radius: 6px;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        border-radius: 6px;
    }
    
    .stError {
        background: #fef2f2;
        color: #991b1b;
        border: 1px solid #fca5a5;
        border-radius: 6px;
    }
    
    .stWarning {
        background: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
        border-radius: 6px;
    }
    
    .stInfo {
        background: #eff6ff;
        color: #1e40af;
        border: 1px solid #93c5fd;
        border-radius: 6px;
    }
    
    /* Metrics */
    div[data-testid="metric-container"] {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 1rem;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: #f9fafb;
        border-radius: 6px;
        padding: 1rem;
    }
    
    /* Clean typography */
    h1, h2, h3, h4, h5, h6 {
        color: #1f2937;
        font-weight: 600;
    }
    
    p, div, span {
        color: #374151;
        line-height: 1.6;
    }
    
    /* Footer */
    .footer {
        background: #1f2937;
        color: white;
        padding: 2rem;
        border-radius: 8px;
        text-align: center;
        margin-top: 2rem;
    }
    
    .footer h3 {
        color: white;
        margin-bottom: 1rem;
    }
    
    .footer p {
        color: #d1d5db;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-title">📄 AI PDF Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload your PDFs and chat with your documents using AI</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Language selection
    language = st.selectbox(
        "🌍 Select Language for Audio", 
        list(LANGUAGE_MAPPING.keys()),
        help="Choose your preferred language for text-to-speech"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Status")
    if "vector_store_ready" in st.session_state and st.session_state.vector_store_ready:
        st.success("PDF Processed Successfully")
    else:
        st.info("Upload PDF to start")
    
    # Stats
    if "chat_history" in st.session_state and st.session_state.chat_history:
        st.metric("Questions Asked", len(st.session_state.chat_history))
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info(f"AI can read your documents and speak answers in {language}")

# File upload section
st.markdown('<div class="upload-container">', unsafe_allow_html=True)
st.markdown("### 📁 Upload Your Documents")
st.markdown("Drag and drop your PDF files here or click to browse")
uploaded_files = st.file_uploader(
    "Choose PDF files", 
    type=["pdf"], 
    accept_multiple_files=True,
    help="Upload one or more PDF files to analyze"
)
st.markdown('</div>', unsafe_allow_html=True)

def extract_text_from_pdfs(files):
    """Extract text from uploaded PDF files"""
    text = ""
    
    try:
        for file_idx, file in enumerate(files):
            file.seek(0)
            
            try:
                pdf_reader = PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text += f"\n\n--- FILE: {file.name} | PAGE {page_num + 1} ---\n{page_text}"
                    except Exception:
                        continue
                        
            except Exception:
                continue
                
        if not text.strip():
            st.error("No text could be extracted from the PDFs. Please check if they contain readable text.")
            return None
            
        return text
        
    except Exception as e:
        st.error(f"Error during PDF processing: {str(e)}")
        return None

def preprocess_text(text):
    """Clean and preprocess text"""
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()
    return text

def create_vector_store(text):
    """Create FAISS vector store from text"""
    if not text or not text.strip():
        st.error("No text provided for vector store creation!")
        return 0
        
    try:
        with st.spinner("Creating AI embeddings..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=300,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            text_chunks = text_splitter.split_text(text)
            text_chunks = [chunk for chunk in text_chunks if len(chunk.strip()) > 50]
            
            if not text_chunks:
                st.error("No valid text chunks created! Please check your PDF content.")
                return 0
            
            try:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )
                
                vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
                vector_store.save_local("faiss_index")
                
                return len(text_chunks)
                
            except Exception as embedding_error:
                st.error(f"Error creating embeddings: {str(embedding_error)}")
                return 0
                
    except Exception as e:
        st.error(f"Error in vector store creation: {str(e)}")
        return 0

def get_conversational_chain():
    """Create conversational chain"""
    try:
        prompt_template = """
        You are an intelligent AI assistant analyzing a document. Use the provided context to answer the question comprehensively.

        INSTRUCTIONS:
        1. If you find relevant information in the context, provide a detailed and helpful answer
        2. If the exact answer isn't available, try to provide related information that might be helpful
        3. Be conversational and helpful in your tone
        4. Only say "Answer is not available in the context" if there is absolutely no relevant information

        Context:
        {context}

        Question: {question}

        Answer:
        """
        
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp", 
            temperature=0.4,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        
        return chain
        
    except Exception as e:
        st.error(f"Error creating conversational chain: {str(e)}")
        return None

def process_user_message(user_input):
    """Process user message"""
    try:
        with st.spinner("Processing your question..."):
            if not os.path.exists("faiss_index"):
                return "Vector store not found. Please upload and process a PDF first."
            
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            
            try:
                vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            except Exception as load_error:
                return f"Error loading vector store: {str(load_error)}. Please reprocess your PDF."
            
            docs = vector_store.similarity_search(user_input, k=6)
            
            chain = get_conversational_chain()
            if not chain:
                return "Error initializing AI model. Please check your API configuration."
            
            try:
                response = chain({"input_documents": docs, "question": user_input}, return_only_outputs=True)
                return response["output_text"]
                
            except Exception as chain_error:
                return f"Error processing question: {str(chain_error)}"
                
    except Exception as e:
        return f"Critical error: {str(e)}"

def speak_text(text, language_code="en"):
    """Convert text to speech"""
    try:
        tts = gTTS(text=text, lang=language_code)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        st.audio(temp_file.name, format="audio/mp3", autoplay=True)
        return True
    except Exception as e:
        st.error(f"Text-to-speech error: {str(e)}")
        return False

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# PDF Processing
if uploaded_files:
    raw_text = extract_text_from_pdfs(uploaded_files)
    
    if raw_text:
        processed_text = preprocess_text(raw_text)
        
        if processed_text:
            chunk_count = create_vector_store(processed_text)
            
            if chunk_count > 0:
                st.session_state.chunk_count = chunk_count
                st.session_state.vector_store_ready = True
                st.success(f"✅ PDFs processed successfully! Created {chunk_count} text chunks.")
            else:
                st.error("Failed to create vector store. Please try again.")

# Chat Interface
st.markdown("### 💬 Chat with your document")

col1, col2 = st.columns([4, 1])

with col1:
    user_input = st.text_input(
        "Ask a question about your document",
        placeholder="What is this document about?",
        label_visibility="collapsed"
    )

with col2:
    ask_button = st.button("Ask", type="primary", use_container_width=True)

# Process question
if ask_button and user_input:
    if "vector_store_ready" not in st.session_state:
        st.warning("Please upload and process a PDF first!")
    else:
        response = process_user_message(user_input)
        st.session_state.chat_history.append((user_input, response))

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    
    for question, answer in reversed(st.session_state.chat_history):
        st.markdown(f'''
        <div class="user-message">
            <strong>You:</strong> {question}
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown(f'''
        <div class="bot-message">
            <strong>AI:</strong> {answer}
        </div>
        ''', unsafe_allow_html=True)

# Action buttons
if st.session_state.chat_history:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔊 Listen", use_container_width=True):
            try:
                lang_code = LANGUAGE_MAPPING[language]
                response_text = st.session_state.chat_history[-1][1]
                
                if speak_text(response_text, lang_code):
                    st.success(f"🎵 Playing audio in {language}")
                else:
                    st.error("Audio playback failed")
                    
            except Exception as e:
                st.error(f"Audio failed: {str(e)}")
    
    with col2:
        if st.button("💾 Export", use_container_width=True):
            chat_text = "\n".join([f"Q: {chat[0]}\nA: {chat[1]}\n{'-'*50}\n" for chat in st.session_state.chat_history])
            st.download_button(
                "📥 Download", 
                chat_text, 
                file_name="chat_history.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    with col3:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.chat_history = []
            if "vector_store_ready" in st.session_state:
                del st.session_state.vector_store_ready
            if "chunk_count" in st.session_state:
                del st.session_state.chunk_count
            st.rerun()

# Feedback
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### ⭐ Feedback")
    
    feedback = st.radio(
        "How was the response?",
        ["⭐ Excellent", "👍 Good", "👎 Needs improvement"],
        horizontal=True
    )
    
    if feedback:
        st.success(f"Thank you for the feedback: {feedback}")

# Footer
st.markdown(
    """
    <div class="footer">
        <h3>📄 AI PDF Chatbot</h3>
        <p>Upload your documents and get intelligent answers powered by AI</p>
    </div>
    """, 
    unsafe_allow_html=True
)
