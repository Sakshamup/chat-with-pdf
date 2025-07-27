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
    # Remove the success message that creates unwanted box
    # st.success("✅ Google API configured successfully!")
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

# Enhanced Professional Red-Orange CSS Theme with Attractive Fonts
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@200;300;400;500;600;700;800;900&family=Inter:wght@200;300;400;500;600;700;800;900&family=Playfair+Display:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@200;300;400;500;600;700;800&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    /* Custom font variables for easy changes */
    :root {
        --font-primary: 'Poppins', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-secondary: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-display: 'Playfair Display', Georgia, serif;
        --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
        --font-modern: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
        
        --text-shadow-soft: 0 2px 8px rgba(0, 0, 0, 0.15);
        --text-shadow-strong: 0 4px 12px rgba(0, 0, 0, 0.3);
        --text-shadow-glow: 0 0 20px rgba(255, 87, 34, 0.4);
    }
    
    /* Main app background - Professional red-orange gradient */
    .stApp {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d1810 25%, #4a1810 50%, #6b2c17 75%, #8b3a1e 100%);
        font-family: var(--font-primary);
        font-weight: 400;
        letter-spacing: 0.3px;
        line-height: 1.6;
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
        font-family: var(--font-display);
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ff5722 0%, #f44336 30%, #e91e63 70%, #d32f2f 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: var(--text-shadow-glow);
        letter-spacing: -0.03em;
        line-height: 1.1;
        font-style: italic;
        position: relative;
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 3px;
        background: linear-gradient(90deg, transparent, #ff5722, transparent);
        border-radius: 2px;
    }
    
    .sub-header {
        font-family: var(--font-modern);
        text-align: center;
        color: #ffffff;
        margin-bottom: 2rem;
        font-size: 1.3rem;
        font-weight: 500;
        background: rgba(255, 87, 34, 0.1);
        padding: 1.5rem 2.5rem;
        border-radius: 18px;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 87, 34, 0.2);
        box-shadow: 0 8px 25px rgba(255, 87, 34, 0.1);
        letter-spacing: 0.5px;
        text-shadow: var(--text-shadow-soft);
        line-height: 1.4;
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
    
    .upload-section h3 {
        font-family: var(--font-modern);
        font-weight: 600;
        letter-spacing: 0.3px;
        color: #ffffff;
        text-shadow: var(--text-shadow-soft);
    }
    
    .upload-section p {
        font-family: var(--font-secondary);
        font-weight: 400;
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.05rem;
        letter-spacing: 0.2px;
    }
    
    /* Removed shimmer animation that was creating empty box */
    
    .user-message {
        font-family: var(--font-secondary);
        background: linear-gradient(135deg, #ff5722 0%, #f4511e 100%);
        color: white;
        padding: 1.4rem 1.8rem;
        border-radius: 18px;
        margin: 1rem 0;
        border-left: 4px solid #d84315;
        box-shadow: 0 8px 25px rgba(255, 87, 34, 0.3);
        font-weight: 500;
        position: relative;
        overflow: hidden;
        font-size: 1.05rem;
        line-height: 1.5;
        letter-spacing: 0.2px;
        text-shadow: var(--text-shadow-soft);
    }
    
    .user-message strong {
        font-family: var(--font-modern);
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        text-shadow: var(--text-shadow-strong);
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
        font-family: var(--font-secondary);
        background: linear-gradient(135deg, #bf360c 0%, #d84315 100%);
        color: white;
        padding: 1.4rem 1.8rem;
        border-radius: 18px;
        margin: 1rem 0;
        border-left: 4px solid #8d2a0e;
        box-shadow: 0 8px 25px rgba(191, 54, 12, 0.3);
        font-weight: 400;
        position: relative;
        overflow: hidden;
        font-size: 1.05rem;
        line-height: 1.6;
        letter-spacing: 0.2px;
        text-shadow: var(--text-shadow-soft);
    }
    
    .bot-message strong {
        font-family: var(--font-modern);
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        color: #ffeb3b;
        text-shadow: var(--text-shadow-strong);
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
        font-family: var(--font-secondary);
    }
    
    .css-1d391kg > div {
        background: transparent !important;
    }
    
    .feature-box {
        font-family: var(--font-modern);
        background: linear-gradient(135deg, rgba(255, 87, 34, 0.15) 0%, rgba(244, 67, 54, 0.1) 100%);
        padding: 1.8rem;
        border-radius: 14px;
        box-shadow: 0 6px 20px rgba(255, 87, 34, 0.15);
        margin: 0.8rem 0;
        text-align: center;
        border: 1px solid rgba(255, 87, 34, 0.25);
        backdrop-filter: blur(15px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        font-weight: 500;
        letter-spacing: 0.3px;
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
        font-family: var(--font-modern);
        background: linear-gradient(135deg, #ff5722 0%, #f4511e 100%);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 1rem 2.5rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 6px 20px rgba(255, 87, 34, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        text-shadow: var(--text-shadow-soft);
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
        font-family: var(--font-secondary);
        border-radius: 14px;
        border: 2px solid rgba(255, 87, 34, 0.3);
        padding: 1rem 1.5rem;
        background: rgba(255, 255, 255, 0.1);
        color: #ffffff;
        font-size: 1.05rem;
        font-weight: 500;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        letter-spacing: 0.3px;
        text-shadow: var(--text-shadow-soft);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.6);
        font-style: italic;
        font-weight: 400;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #ff5722;
        box-shadow: 0 0 20px rgba(255, 87, 34, 0.3);
        background: rgba(255, 255, 255, 0.15);
        outline: none;
    }
    
    /* Professional Section Headers */
    h3 {
        font-family: var(--font-modern);
        color: #ffffff;
        font-weight: 700;
        font-size: 1.4rem;
        background: linear-gradient(135deg, rgba(255, 87, 34, 0.2) 0%, rgba(244, 67, 54, 0.1) 100%);
        padding: 1rem 1.8rem;
        border-radius: 14px;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 87, 34, 0.25);
        box-shadow: 0 4px 15px rgba(255, 87, 34, 0.1);
        margin-bottom: 1rem;
        letter-spacing: 0.5px;
        text-shadow: var(--text-shadow-soft);
        text-transform: capitalize;
    }
    
    /* Radio button container */
    .stRadio > div {
        font-family: var(--font-secondary);
        background: rgba(255, 87, 34, 0.1);
        padding: 1.5rem;
        border-radius: 14px;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 87, 34, 0.2);
        box-shadow: 0 4px 15px rgba(255, 87, 34, 0.1);
    }
    
    .stRadio label {
        font-weight: 500;
        letter-spacing: 0.2px;
        color: #ffffff;
        text-shadow: var(--text-shadow-soft);
    }
    
    /* Metrics styling */
    .css-1r6slb0 {
        font-family: var(--font-modern);
        background: linear-gradient(135deg, rgba(255, 87, 34, 0.15) 0%, rgba(244, 67, 54, 0.1) 100%);
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 6px 20px rgba(255, 87, 34, 0.15);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 87, 34, 0.2);
    }
    
    /* Enhanced File uploader styling */
    .css-1cpxqw2, 
    section[data-testid="stFileUploader"],
    .css-1cpxqw2 > div,
    div[data-testid="stFileUploader"] {
        font-family: var(--font-secondary);
        background: linear-gradient(135deg, rgba(255, 87, 34, 0.1) 0%, rgba(244, 67, 54, 0.05) 100%) !important;
        border-radius: 18px !important;
        border: 2px dashed rgba(255, 87, 34, 0.4) !important;
        padding: 2.5rem !important;
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
        letter-spacing: 0.2px !important;
        text-shadow: var(--text-shadow-soft) !important;
    }
    
    /* Browse files button styling */
    div[data-testid="stFileUploader"] button {
        font-family: var(--font-modern) !important;
        background: linear-gradient(135deg, #ff5722 0%, #f4511e 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 1rem 2rem !important;
        font-weight: 700 !important;
        box-shadow: 0 6px 20px rgba(255, 87, 34, 0.3) !important;
        transition: all 0.3s ease !important;
        letter-spacing: 0.5px !important;
        text-transform: uppercase !important;
        text-shadow: var(--text-shadow-soft) !important;
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
        padding: 1.5rem;
        border-radius: 15px;
        font-family: var(--font-secondary);
    }
    
    /* Sidebar text styling */
    section[data-testid="stSidebar"] h3 {
        font-family: var(--font-modern);
        background: linear-gradient(135deg, rgba(255, 87, 34, 0.2) 0%, rgba(244, 67, 54, 0.15) 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 87, 34, 0.25);
        color: #ffffff;
        box-shadow: 0 4px 15px rgba(255, 87, 34, 0.1);
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        text-shadow: var(--text-shadow-soft);
    }
    
    /* Sidebar metrics styling */
    section[data-testid="stSidebar"] .css-1r6slb0 {
        font-family: var(--font-modern);
        background: linear-gradient(135deg, rgba(255, 87, 34, 0.15) 0%, rgba(244, 67, 54, 0.1) 100%);
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 15px rgba(255, 87, 34, 0.1);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 87, 34, 0.2);
    }
    
    /* Sidebar info box styling */
    section[data-testid="stSidebar"] .stAlert {
        font-family: var(--font-secondary);
        background: linear-gradient(135deg, rgba(255, 87, 34, 0.2) 0%, rgba(244, 67, 54, 0.15) 100%);
        color: #ffffff;
        border-radius: 12px;
        border: 1px solid rgba(255, 87, 34, 0.25);
        backdrop-filter: blur(15px);
        box-shadow: 0 4px 15px rgba(255, 87, 34, 0.1);
        font-weight: 500;
        letter-spacing: 0.2px;
        text-shadow: var(--text-shadow-soft);
    }
    
    /* Sidebar markdown text */
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] .stMarkdown {
        font-family: var(--font-secondary);
        color: #ffffff;
        font-weight: 400;
        letter-spacing: 0.2px;
        line-height: 1.5;
        text-shadow: var(--text-shadow-soft);
    }

    /* Enhanced Success, Error, and Warning messages */
    .stSuccess {
        font-family: var(--font-secondary);
        background: linear-gradient(135deg, #4caf50 0%, #388e3c 100%);
        color: white;
        border-radius: 14px;
        border: 1px solid rgba(76, 175, 80, 0.3);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.2);
        font-weight: 500;
        letter-spacing: 0.2px;
        text-shadow: var(--text-shadow-soft);
    }
    
    .stError {
        font-family: var(--font-secondary);
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        color: white;
        border-radius: 14px;
        border: 1px solid rgba(244, 67, 54, 0.3);
        box-shadow: 0 6px 20px rgba(244, 67, 54, 0.2);
        font-weight: 500;
        letter-spacing: 0.2px;
        text-shadow: var(--text-shadow-soft);
    }
    
    .stWarning {
        font-family: var(--font-secondary);
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        color: white;
        border-radius: 14px;
        border: 1px solid rgba(255, 152, 0, 0.3);
        box-shadow: 0 6px 20px rgba(255, 152, 0, 0.2);
        font-weight: 500;
        letter-spacing: 0.2px;
        text-shadow: var(--text-shadow-soft);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        font-family: var(--font-secondary);
        background: rgba(255, 255, 255, 0.1);
        color: #ffffff;
        border-radius: 12px;
        border: 2px solid rgba(255, 87, 34, 0.3);
        backdrop-filter: blur(10px);
        font-weight: 500;
        letter-spacing: 0.2px;
    }
    
    .stSelectbox label {
        font-family: var(--font-modern);
        color: #ffffff;
        font-weight: 600;
        letter-spacing: 0.3px;
        text-shadow: var(--text-shadow-soft);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 87, 34, 0.1);
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #ff5722 0%, #f4511e 100%);
        border-radius: 6px;
        box-shadow: 0 2px 8px rgba(255, 87, 34, 0.3);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #f4511e 0%, #d84315 100%);
    }
    
    /* Loading spinner animation */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Enhanced font smoothing */
    * {
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        text-rendering: optimizeLegibility;
    }
    
    /* Responsive design improvements */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .sub-header {
            font-size: 1.1rem;
            padding: 1.2rem 1.8rem;
        }
        
        .upload-section {
            padding: 1.8rem;
        }
        
        .feature-box {
            padding: 1.5rem;
        }
        
        .user-message, .bot-message {
            padding: 1.2rem 1.5rem;
            font-size: 1rem;
        }
    }
    
    /* Additional font weight utilities */
    .font-light { font-weight: 300; }
    .font-normal { font-weight: 400; }
    .font-medium { font-weight: 500; }
    .font-semibold { font-weight: 600; }
    .font-bold { font-weight: 700; }
    .font-extrabold { font-weight: 800; }
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
    ask_button = st.button("Ask", type="primary", use_container_width=True)

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
    st.markdown("### 💭 Conversation History")
    
    for i, (question, answer) in enumerate(reversed(st.session_state.chat_history)):
        # User message
        st.markdown(f'''
        <div class="user-message">
            <strong>You:</strong> {question}
        </div>
        ''', unsafe_allow_html=True)
        
        # Bot message
        st.markdown(f'''
        <div class="bot-message">
            <strong>AI:</strong> {answer}
        </div>
        ''', unsafe_allow_html=True)
        
        if i < len(st.session_state.chat_history) - 1:
            st.markdown("---")

# Feature buttons
if st.session_state.chat_history:
    st.markdown("### 🛠️ Actions")
    
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
                    st.success(f"🎵 Audio ready in {language}")
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
    st.markdown("### ⭐ Feedback")
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        feedback = st.radio(
            "How was the response?",
            ["⭐ Excellent", "👍 Good", "👎 Needs improvement"],
            horizontal=True
        )
    
    with col2:
        if feedback:
            st.success(f"✅ Thank you for the feedback: {feedback}")

# Enhanced Footer with beautiful typography
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #ffffff; padding: 3rem; background: linear-gradient(135deg, #ff5722 0%, #f4511e 30%, #e91e63 70%, #d32f2f 100%); border-radius: 20px; margin-top: 2rem; box-shadow: 0 15px 40px rgba(255, 87, 34, 0.4); position: relative; overflow: hidden; font-family: var(--font-display);'>
        <div style='position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: radial-gradient(circle at 30% 70%, rgba(255, 255, 255, 0.15) 0%, transparent 60%), radial-gradient(circle at 70% 30%, rgba(255, 255, 255, 0.1) 0%, transparent 50%); pointer-events: none;'></div>
        <h3 style='margin: 0; font-size: 2.2rem; font-weight: 800; text-shadow: 0 4px 12px rgba(0,0,0,0.4); position: relative; z-index: 1; font-style: italic; letter-spacing: -0.02em;'>📄 AI PDF Chatbot</h3>
        <p style='font-size: 1.3rem; margin: 1.2rem 0; font-weight: 600; position: relative; z-index: 1; font-family: var(--font-modern); letter-spacing: 0.5px; text-transform: uppercase;'>
            <strong>Upload PDF → Ask Questions → Get Intelligent Answers</strong>
        </p>
        <p style='font-size: 1.1rem; opacity: 0.95; margin: 0; position: relative; z-index: 1; font-family: var(--font-secondary); font-weight: 400; font-style: italic; letter-spacing: 0.3px;'>
            Experience next-generation document analysis with professional AI assistance
        </p>
    </div>
    """, 
    unsafe_allow_html=True)
