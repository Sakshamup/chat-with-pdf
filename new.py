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

# Normalized CSS with same colors but reduced animations
st.markdown("""
<style>
    /* Main app background with static gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
    }
    
    /* Main page container with glass effect */
    .main .block-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .sub-header {
        text-align: center;
        color: #ffffff;
        margin-bottom: 2rem;
        font-size: 1.1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    .upload-section {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #a8edea 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px dashed #ff6b6b;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.2);
    }
    
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #ff6b6b;
        box-shadow: 0 3px 10px rgba(102, 126, 234, 0.3);
    }
    
    .bot-message {
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #feca57;
        box-shadow: 0 3px 10px rgba(78, 205, 196, 0.3);
    }
    
    /* Sidebar with gradient */
    .css-1d391kg, .css-18e3th9, section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .css-1d391kg > div {
        background: transparent !important;
    }
    
    .feature-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 50%, #a8edea 100%);
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(252, 182, 159, 0.3);
        margin: 0.5rem 0;
        text-align: center;
        border: 1px solid #ff9ff3;
    }
    
    .feature-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(252, 182, 159, 0.4);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.6rem 1.8rem;
        font-weight: 600;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        background: linear-gradient(45deg, #4ecdc4, #ff6b6b, #feca57);
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        border-radius: 20px;
        border: 2px solid transparent;
        background: linear-gradient(white, white) padding-box,
                   linear-gradient(45deg, #ff6b6b, #4ecdc4, #feca57) border-box;
        padding: 0.6rem 1rem;
    }
    
    /* Success/warning message styling */
    .success {
        background: linear-gradient(135deg, #4ecdc4, #44a08d);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 3px 10px rgba(78, 205, 196, 0.3);
    }
    
    /* Section headers */
    h3 {
        color: #ffffff;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        font-weight: 600;
        background: rgba(255, 255, 255, 0.1);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        backdrop-filter: blur(10px);
    }
    
    /* Radio button container */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.2);
        padding: 1rem;
        border-radius: 12px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Metrics styling */
    .css-1r6slb0 {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
    }
    
    /* File uploader styling */
    .css-1cpxqw2, 
    section[data-testid="stFileUploader"],
    .css-1cpxqw2 > div,
    div[data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 100%) !important;
        border-radius: 15px !important;
        border: 2px dashed rgba(255, 255, 255, 0.8) !important;
        padding: 1.5rem !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* File uploader text styling */
    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] p,
    div[data-testid="stFileUploader"] small {
        color: #ffffff !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5) !important;
        font-weight: 500 !important;
    }
    
    /* Browse files button styling */
    div[data-testid="stFileUploader"] button {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4) !important;
        color: white !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.2s ease !important;
    }
    
    div[data-testid="stFileUploader"] button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
        background: linear-gradient(45deg, #4ecdc4, #ff6b6b) !important;
    }
    
    /* File upload icon styling */
    div[data-testid="stFileUploader"] svg {
        color: #ffffff !important;
        filter: drop-shadow(1px 1px 2px rgba(0,0,0,0.5)) !important;
    }
    
    /* Sidebar content styling */
    .css-18e3th9 {
        background: transparent !important;
        padding: 1rem;
        border-radius: 12px;
    }
    
    /* Sidebar text styling */
    section[data-testid="stSidebar"] h3 {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.6rem 1rem;
        border-radius: 10px;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: #ffffff;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    /* Sidebar metrics styling */
    section[data-testid="stSidebar"] .css-1r6slb0 {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Sidebar info box styling */
    section[data-testid="stSidebar"] .stAlert {
        background: rgba(255, 255, 255, 0.2);
        color: #ffffff;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar markdown text */
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] .stMarkdown {
        color: #ffffff;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
</style>
""", unsafe_allow_html=True)

# Main title with normalized styling
st.markdown('<h1 class="main-header">🌟 AI PDF Chatbot 🌟</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">✨ Intelligent Document Analysis with Enhanced Answer Retrieval ✨</p>', unsafe_allow_html=True)

# Sidebar with styling
with st.sidebar:
    st.markdown("### 🎨 Settings")
    
    # Language selection for text-to-speech
    language = st.selectbox(
        "🌍 Select Language for Audio", 
        list(LANGUAGE_MAPPING.keys()),
        help="Choose your preferred language for text-to-speech playback"
    )
    
    st.markdown("---")
    st.markdown("### 🎯 Status")
    if "vector_store_ready" in st.session_state and st.session_state.vector_store_ready:
        st.markdown("🟢 **PDF Processed Successfully!**")
    else:
        st.markdown("🟡 **Upload PDF to start!**")
    
    # Quick stats
    if "chat_history" in st.session_state and st.session_state.chat_history:
        st.metric("💬 Total Questions", len(st.session_state.chat_history))
    
    # Info section
    st.markdown("---")
    st.markdown("### 💡 Info")
    st.info(f"Enhanced AI can read your documents and speak answers in {language}!")

# File upload section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("### 🚀 Upload Your Documents")
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
        "process": ["procedure", "method", "steps", "approach"],
        "result": ["outcome", "finding", "conclusion", "effect"]
    }
    
    query_lower = query.lower()
    for key, synonyms in common_synonyms.items():
        if key in query_lower:
            enhanced_terms.extend(synonyms)
    
    return " ".join(enhanced_terms)

def process_user_message(user_input):
    """Process user message with enhanced error handling"""
    try:
        with st.spinner("🤔 Processing your question..."):
            # Check if vector store exists
            if not os.path.exists("faiss_index"):
                return "❌ Vector store not found. Please upload and process a PDF first."
            
            # Initialize embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            
            # Load vector store
            try:
                vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            except Exception as load_error:
                return f"❌ Error loading vector store: {str(load_error)}. Please reprocess your PDF."
            
            # Enhanced query
            enhanced_query = enhance_query(user_input)
            
            # Get more documents for better context (fixed at 6)
            num_docs = 6
            docs = vector_store.similarity_search(enhanced_query, k=num_docs)
        
            # Also try with the original query if enhanced query doesn't yield good results
            if len(docs) < num_docs:
                additional_docs = vector_store.similarity_search(user_input, k=num_docs)
                # Combine and deduplicate
                all_docs = docs + additional_docs
                seen_content = set()
                unique_docs = []
                for doc in all_docs:
                    if doc.page_content not in seen_content:
                        unique_docs.append(doc)
                        seen_content.add(doc.page_content)
                docs = unique_docs[:num_docs]
            
            # If still no good matches, try with more relaxed search
            if len(docs) < 3:
                # Try searching with individual words from the query
                words = user_input.split()
                for word in words:
                    if len(word) > 3:  # Only search meaningful words
                        word_docs = vector_store.similarity_search(word, k=2)
                        docs.extend(word_docs)
            
            # Get conversational chain
            chain = get_conversational_chain()
            if not chain:
                return "❌ Error initializing AI model. Please check your API configuration."
            
            try:
                response = chain({"input_documents": docs, "question": user_input}, return_only_outputs=True)
                answer = response["output_text"]
                
                # If we still get a "not available" response, try a fallback approach
                if "not available in the context" in answer.lower() or "cannot find" in answer.lower():
                    # Try a more general search
                    general_docs = vector_store.similarity_search(user_input, k=10)
                    if general_docs:
                        fallback_response = chain({"input_documents": general_docs, "question": f"Based on the available information, what can you tell me about: {user_input}"}, return_only_outputs=True)
                        fallback_answer = fallback_response["output_text"]
                        if "not available" not in fallback_answer.lower():
                            answer = fallback_answer
                
                return answer
                
            except Exception as chain_error:
                return f"❌ Error processing question: {str(chain_error)}"
                
    except Exception as e:
        return f"❌ Critical error: {str(e)}"

def speak_text(text, language_code="en"):
    """Convert text to speech and play it"""
    try:
        tts = gTTS(text=text, lang=language_code)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        st.audio(temp_file.name, format="audio/mp3", autoplay=True)
        return True
    except Exception as e:
        st.error(f"❌ Text-to-speech error: {str(e)}")
        return False

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# PDF Processing
if uploaded_files:
    # Extract text from PDFs
    raw_text = extract_text_from_pdfs(uploaded_files)
    
    if raw_text:
        # Preprocess text
        processed_text = preprocess_text(raw_text)
        
        if processed_text:
            # Create vector store and get chunk count
            chunk_count = create_vector_store(processed_text)
            
            if chunk_count > 0:
                # Store the chunk count and set ready flag
                st.session_state.chunk_count = chunk_count
                st.session_state.vector_store_ready = True
                st.success(f"✅ PDFs processed successfully!")
            else:
                st.error("❌ Failed to create vector store. Please try again or check your API configuration.")
        else:
            st.error("❌ Failed to preprocess text. Please check your PDF files.")
    else:
        st.error("❌ Failed to extract text from PDFs. Please ensure your PDFs contain readable text.")

# Chat Interface
st.markdown("### 💬 Ask Questions")

# Create columns for better layout
col1, col2 = st.columns([4, 1])

with col1:
    user_input = st.text_input(
        "What would you like to know about your document?",
        placeholder="e.g., What is the main topic of this document?",
        label_visibility="collapsed"
    )

with col2:
    ask_button = st.button("🚀 Ask", type="primary", use_container_width=True)

# Process question
if ask_button and user_input:
    if "vector_store_ready" not in st.session_state:
        st.warning("⚠️ Please upload and process a PDF first!")
    else:
        # Get the response
        response = process_user_message(user_input)
        
        # Store the response
        st.session_state.chat_history.append((user_input, response))

# Display chat history
if st.session_state.chat_history:
    st.markdown("### 🗨️ Conversation History")
    
    for i, (question, answer) in enumerate(reversed(st.session_state.chat_history)):
        # User message
        st.markdown(f'''
        <div class="user-message">
            <strong>🙋‍♀️ You:</strong> {question}
        </div>
        ''', unsafe_allow_html=True)
        
        # Bot message
        st.markdown(f'''
        <div class="bot-message">
            <strong>🤖 AI:</strong> {answer}
        </div>
        ''', unsafe_allow_html=True)
        
        if i < len(st.session_state.chat_history) - 1:
            st.markdown("---")

# Feature buttons
if st.session_state.chat_history:
    st.markdown("### 🛠️ Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        if st.button("🔊 Listen", use_container_width=True):
            try:
                # Get the language code for the selected language
                lang_code = LANGUAGE_MAPPING[language]
                response_text = st.session_state.chat_history[-1][1]
                
                # Use the speak_text function with selected language
                if speak_text(response_text, lang_code):
                    st.success(f"🎵 Audio ready in {language}!")
                else:
                    st.error(f"Audio failed for {language}")
                    
            except Exception as e:
                st.error(f"Audio failed: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        if st.button("💾 Export", use_container_width=True):
            chat_text = "\n".join([f"Q: {chat[0]}\nA: {chat[1]}\n{'-'*50}\n" for chat in st.session_state.chat_history])
            st.download_button(
                "📥 Download", 
                chat_text, 
                file_name="chat_history.txt",
                mime="text/plain",
                use_container_width=True
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.chat_history = []
            if "vector_store_ready" in st.session_state:
                del st.session_state.vector_store_ready
            if "chunk_count" in st.session_state:
                del st.session_state.chunk_count
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# Feedback section
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### ⭐ Rate the Response")
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        feedback = st.radio(
            "How was the response?",
            ["⭐ Excellent!", "👍 Good!", "👎 Could be better"],
            horizontal=True
        )
    
    with col2:
        if feedback:
            st.success(f"✅ Thank you for the feedback: {feedback}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #ffffff; padding: 1.5rem; background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #feca57, #ff9ff3); border-radius: 12px; margin-top: 2rem; box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);'>
        <h3>🌈 Enhanced PDF Chatbot</h3>
        <p style='font-size: 1rem; margin-top: 1rem;'>
            🚀 <strong>Upload PDF → Ask Questions → Get Answers!</strong> 🌟
        </p>
        <p style='font-size: 0.9rem; opacity: 0.9;'>
            Experience intelligent document analysis with AI 🤖
        </p>
    </div>
    """, 
    unsafe_allow_html=True)
