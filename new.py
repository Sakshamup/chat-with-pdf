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

headers = {
    "authorization": os.getenv("GOOGLE_API_KEY"),
    "content-type": "application/json"
}

# Page config with custom styling
st.set_page_config(
    page_title="AI PDF Chatbot", 
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Red-Orange CSS Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main app background - Professional red-orange gradient */
    .stApp {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d1810 25%, #4a1810 50%, #6b2c17 75%, #8b3a1e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main page container with sophisticated glass effect */
    .main .block-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 87, 34, 0.2);
        padding: 2.5rem;
        margin: 1rem;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.25), 
                    0 0 0 1px rgba(255, 87, 34, 0.1);
        position: relative;
    }
    
    /* Add subtle animated background pattern */
    .main .block-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 20% 80%, rgba(244, 67, 54, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 80% 20%, rgba(255, 87, 34, 0.1) 0%, transparent 50%);
        border-radius: 20px;
        pointer-events: none;
        z-index: -1;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ff5722 0%, #f44336 50%, #d32f2f 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 4px 8px rgba(244, 67, 54, 0.3);
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        text-align: center;
        color: #ffffff;
        margin-bottom: 2rem;
        font-size: 1.2rem;
        font-weight: 400;
        background: rgba(255, 87, 34, 0.1);
        padding: 1.2rem 2rem;
        border-radius: 15px;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 87, 34, 0.2);
        box-shadow: 0 8px 25px rgba(255, 87, 34, 0.1);
    }
    
    .upload-section {
        background: linear-gradient(135deg, rgba(255, 87, 34, 0.1) 0%, rgba(244, 67, 54, 0.05) 100%);
        padding: 2rem;
        border-radius: 16px;
        border: 2px dashed rgba(255, 87, 34, 0.4);
        margin: 1.5rem 0;
        box-shadow: 0 8px 30px rgba(255, 87, 34, 0.15);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .upload-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 87, 34, 0.1), transparent);
        transform: rotate(45deg);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .user-message {
        background: linear-gradient(135deg, #ff5722 0%, #f4511e 100%);
        color: white;
        padding: 1.2rem 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #d84315;
        box-shadow: 0 8px 25px rgba(255, 87, 34, 0.3);
        font-weight: 500;
        position: relative;
        overflow: hidden;
    }
    
    .user-message::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transform: translateX(-100%);
        transition: transform 0.6s;
    }
    
    .user-message:hover::before {
        transform: translateX(100%);
    }
    
    .bot-message {
        background: linear-gradient(135deg, #bf360c 0%, #d84315 100%);
        color: white;
        padding: 1.2rem 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #8d2a0e;
        box-shadow: 0 8px 25px rgba(191, 54, 12, 0.3);
        font-weight: 500;
        position: relative;
        overflow: hidden;
    }
    
    .bot-message::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transform: translateX(-100%);
        transition: transform 0.6s;
    }
    
    .bot-message:hover::before {
        transform: translateX(100%);
    }
    
    /* Professional Sidebar */
    .css-1d391kg, .css-18e3th9, section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(26, 26, 26, 0.95) 0%, rgba(45, 24, 16, 0.95) 50%, rgba(74, 24, 16, 0.95) 100%);
        backdrop-filter: blur(20px);
        border-radius: 15px;
        border: 1px solid rgba(255, 87, 34, 0.2);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
    }
    
    .css-1d391kg > div {
        background: transparent !important;
    }
    
    .feature-box {
        background: linear-gradient(135deg, rgba(255, 87, 34, 0.15) 0%, rgba(244, 67, 54, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(255, 87, 34, 0.15);
        margin: 0.8rem 0;
        text-align: center;
        border: 1px solid rgba(255, 87, 34, 0.25);
        backdrop-filter: blur(15px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .feature-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .feature-box:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 12px 30px rgba(255, 87, 34, 0.25);
        background: linear-gradient(135deg, rgba(255, 87, 34, 0.25) 0%, rgba(244, 67, 54, 0.2) 100%);
        border-color: rgba(255, 87, 34, 0.4);
    }
    
    .feature-box:hover::before {
        left: 100%;
    }
    
    /* Professional Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #ff5722 0%, #f4511e 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 0 6px 20px rgba(255, 87, 34, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        letter-spacing: 0.5px;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(255, 87, 34, 0.4);
        background: linear-gradient(135deg, #f4511e 0%, #d84315 100%);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:active {
        transform: translateY(0px);
        box-shadow: 0 4px 15px rgba(255, 87, 34, 0.3);
    }
    
    /* Enhanced Input Field Styling */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid rgba(255, 87, 34, 0.3);
        padding: 0.8rem 1.2rem;
        background: rgba(255, 255, 255, 0.1);
        color: #ffffff;
        font-size: 1rem;
        font-weight: 500;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.6);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #ff5722;
        box-shadow: 0 0 20px rgba(255, 87, 34, 0.3);
        background: rgba(255, 255, 255, 0.15);
        outline: none;
    }
    
    /* Professional Section Headers */
    h3 {
        color: #ffffff;
        font-weight: 600;
        font-size: 1.3rem;
        background: linear-gradient(135deg, rgba(255, 87, 34, 0.2) 0%, rgba(244, 67, 54, 0.1) 100%);
        padding: 0.8rem 1.5rem;
        border-radius: 12px;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 87, 34, 0.25);
        box-shadow: 0 4px 15px rgba(255, 87, 34, 0.1);
        margin-bottom: 1rem;
    }
    
    /* Radio button container */
    .stRadio > div {
        background: rgba(255, 87, 34, 0.1);
        padding: 1.2rem;
        border-radius: 12px;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 87, 34, 0.2);
        box-shadow: 0 4px 15px rgba(255, 87, 34, 0.1);
    }
    
    /* Metrics styling */
    .css-1r6slb0 {
        background: linear-gradient(135deg, rgba(255, 87, 34, 0.15) 0%, rgba(244, 67, 54, 0.1) 100%);
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 6px 20px rgba(255, 87, 34, 0.15);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 87, 34, 0.2);
    }
    
    /* Enhanced File uploader styling */
    .css-1cpxqw2, 
    section[data-testid="stFileUploader"],
    .css-1cpxqw2 > div,
    div[data-testid="stFileUploader"] {
        background: linear-gradient(135deg, rgba(255, 87, 34, 0.1) 0%, rgba(244, 67, 54, 0.05) 100%) !important;
        border-radius: 16px !important;
        border: 2px dashed rgba(255, 87, 34, 0.4) !important;
        padding: 2rem !important;
        box-shadow: 0 8px 25px rgba(255, 87, 34, 0.15) !important;
        transition: all 0.3s ease !important;
        position: relative !important;
    }
    
    div[data-testid="stFileUploader"]:hover {
        border-color: rgba(255, 87, 34, 0.6) !important;
        background: linear-gradient(135deg, rgba(255, 87, 34, 0.15) 0%, rgba(244, 67, 54, 0.1) 100%) !important;
        transform: translateY(-2px) !important;
    }
    
    /* File uploader text styling */
    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] p,
    div[data-testid="stFileUploader"] small {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    /* Browse files button styling */
    div[data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #ff5722 0%, #f4511e 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.8rem 1.8rem !important;
        font-weight: 600 !important;
        box-shadow: 0 6px 20px rgba(255, 87, 34, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="stFileUploader"] button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(255, 87, 34, 0.4) !important;
        background: linear-gradient(135deg, #f4511e 0%, #d84315 100%) !important;
    }
    
    /* File upload icon styling */
    div[data-testid="stFileUploader"] svg {
        color: #ffffff !important;
        filter: drop-shadow(0 2px 4px rgba(255, 87, 34, 0.3)) !important;
    }
    
    /* Sidebar content styling */
    .css-18e3th9 {
        background: transparent !important;
        padding: 1.2rem;
        border-radius: 15px;
    }
    
    /* Sidebar text styling */
    section[data-testid="stSidebar"] h3 {
        background: linear-gradient(135deg, rgba(255, 87, 34, 0.2) 0%, rgba(244, 67, 54, 0.15) 100%);
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 87, 34, 0.25);
        color: #ffffff;
        box-shadow: 0 4px 15px rgba(255, 87, 34, 0.1);
    }
    
    /* Sidebar metrics styling */
    section[data-testid="stSidebar"] .css-1r6slb0 {
        background: linear-gradient(135deg, rgba(255, 87, 34, 0.15) 0%, rgba(244, 67, 54, 0.1) 100%);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(255, 87, 34, 0.1);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 87, 34, 0.2);
    }
    
    /* Sidebar info box styling */
    section[data-testid="stSidebar"] .stAlert {
        background: linear-gradient(135deg, rgba(255, 87, 34, 0.2) 0%, rgba(244, 67, 54, 0.15) 100%);
        color: #ffffff;
        border-radius: 10px;
        border: 1px solid rgba(255, 87, 34, 0.25);
        backdrop-filter: blur(15px);
        box-shadow: 0 4px 15px rgba(255, 87, 34, 0.1);
    }
    
    /* Sidebar markdown text */
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] .stMarkdown {
        color: #ffffff;
        font-weight: 400;
    }

    /* Enhanced Success, Error, and Warning messages */
    .stSuccess {
        background: linear-gradient(135deg, #4caf50 0%, #388e3c 100%);
        color: white;
        border-radius: 12px;
        border: 1px solid rgba(76, 175, 80, 0.3);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.2);
    }
    
    .stError {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        color: white;
        border-radius: 12px;
        border: 1px solid rgba(244, 67, 54, 0.3);
        box-shadow: 0 6px 20px rgba(244, 67, 54, 0.2);
    }
    
    .stWarning {
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        color: white;
        border-radius: 12px;
        border: 1px solid rgba(255, 152, 0, 0.3);
        box-shadow: 0 6px 20px rgba(255, 152, 0, 0.2);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 87, 34, 0.1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #ff5722 0%, #f4511e 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #f4511e 0%, #d84315 100%);
    }
    
    /* Loading spinner animation */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Responsive design improvements */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.2rem;
        }
        
        .sub-header {
            font-size: 1rem;
            padding: 1rem 1.5rem;
        }
        
        .upload-section {
            padding: 1.5rem;
        }
        
        .feature-box {
            padding: 1.2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">📄 AI PDF Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Professional Document Analysis with Enhanced AI Intelligence</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Language selection for text-to-speech
    language = st.selectbox(
        "🌍 Select Language for Audio", 
        list(LANGUAGE_MAPPING.keys()),
        help="Choose your preferred language for text-to-speech playback"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Status")
    if "vector_store_ready" in st.session_state and st.session_state.vector_store_ready:
        st.markdown("🟢 **PDF Processed Successfully**")
    else:
        st.markdown("🟡 **Upload PDF to start**")
    
    # Quick stats
    if "chat_history" in st.session_state and st.session_state.chat_history:
        st.metric("💬 Questions Asked", len(st.session_state.chat_history))
    
    # Info section
    st.markdown("---")
    st.markdown("### ℹ️ Information")
    st.info(f"AI can read your documents and speak answers in {language}")

# File upload section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("### 📁 Upload Your Documents")
st.markdown("**Drag & drop your PDF files here or click to browse**")
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
    total_pages = 0
    
    try:
        for file_idx, file in enumerate(files):
            # Reset file pointer to beginning
            file.seek(0)
            
            try:
                pdf_reader = PdfReader(file)
                file_pages = len(pdf_reader.pages)
                total_pages += file_pages
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():  # Only add non-empty pages
                            text += f"\n\n--- FILE: {file.name} | PAGE {page_num + 1} ---\n{page_text}"
                            
                    except Exception as page_error:
                        continue
                        
            except Exception as file_error:
                continue
                
        if not text.strip():
            st.error("❌ No text could be extracted from the uploaded PDFs. Please check if they contain readable text.")
            return None
            
        return text
        
    except Exception as e:
        st.error(f"❌ Critical error during PDF processing: {str(e)}")
        return None

def preprocess_text(text):
    # Improved text preprocessing
    # Remove extra whitespace but preserve paragraph breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces and tabs
    text = text.strip()
    return text

def create_vector_store(text):
    """Create FAISS vector store from text"""
    if not text or not text.strip():
        st.error("❌ No text provided for vector store creation!")
        return 0
        
    try:
        with st.spinner("🔄 Creating AI embeddings..."):
            # Improved text splitting with better parameters
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,  # Smaller chunks for better precision
                chunk_overlap=300,  # More overlap for better context preservation
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]  # Better separation logic
            )
            
            text_chunks = text_splitter.split_text(text)
            
            # Filter out very short chunks that might not be useful
            text_chunks = [chunk for chunk in text_chunks if len(chunk.strip()) > 50]
            
            if not text_chunks:
                st.error("❌ No valid text chunks created! Please check your PDF content.")
                return 0
            
            # Initialize embeddings with error handling
            try:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )
                
                vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
                vector_store.save_local("faiss_index")
                
                # Return the actual chunk count
                return len(text_chunks)
                
            except Exception as embedding_error:
                st.error(f"❌ Error creating embeddings: {str(embedding_error)}")
                st.error("🔍 This might be due to API key issues or network problems.")
                return 0
                
    except Exception as e:
        st.error(f"❌ Error in vector store creation: {str(e)}")
        return 0

def get_conversational_chain():
    """Create conversational chain with error handling"""
    try:
        # Enhanced prompt template with better instructions
        prompt_template = """
        You are an intelligent AI assistant analyzing a document. Use the provided context to answer the question as comprehensively as possible.

        INSTRUCTIONS:
        1. If you find relevant information in the context, provide a detailed and helpful answer
        2. If the exact answer isn't available, try to provide related information that might be helpful
        3. If you can make reasonable inferences from the available information, do so while indicating they are inferences
        4. Only say "Answer is not available in the context" if there is absolutely no relevant information
        5. Be conversational and helpful in your tone
        6. If appropriate, suggest what additional information might be needed

        Context:
        {context}

        Question: {question}

        Helpful Answer:
        """
        
        # Initialize model with error handling
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp", 
            temperature=0.4,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        
        return chain
        
    except Exception as e:
        st.error(f"❌ Error creating conversational chain: {str(e)}")
        return None

def enhance_query(query):
    """Enhance user query with synonyms and related terms"""
    # Simple query enhancement - you can make this more sophisticated
    enhanced_terms = []
    
    # Add the original query
    enhanced_terms.append(query)
    
    # Add variations (you can expand this with a proper thesaurus)
    common_synonyms = {
        "summary": ["overview", "abstract", "conclusion", "main points"],
        "benefits": ["advantages", "positives", "pros", "value"],
        "problems": ["issues", "challenges", "difficulties", "concerns"],
        "cost": ["price", "expense", "budget", "financial"],
        "process": ["procedure", "method", "steps",
