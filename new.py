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

# Enhanced Professional CSS Theme with Better Typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Global font and text improvements */
    * {
        font-family: 'Poppins', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        letter-spacing: 0.3px;
    }
    
    /* Main app background - Professional dark teal gradient */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 25%, #2c5f5f 50%, #0ABAB5 75%, #14b8a6 100%);
        font-family: 'Poppins', sans-serif;
        min-height: 100vh;
        line-height: 1.6;
    }
    
    /* Main page container with sophisticated glass effect */
    .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(25px);
        border-radius: 20px;
        border: 1px solid rgba(10, 186, 181, 0.3);
        padding: 2.5rem;
        margin: 1rem;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3), 
                    0 0 0 1px rgba(10, 186, 181, 0.2);
        position: relative;
        color: #1a1a1a;
    }
    
    /* Add subtle animated background pattern */
    .main .block-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 20% 80%, rgba(10, 186, 181, 0.05) 0%, transparent 50%),
                    radial-gradient(circle at 80% 20%, rgba(255, 237, 243, 0.05) 0%, transparent 50%);
        border-radius: 20px;
        pointer-events: none;
        z-index: -1;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0ABAB5 0%, #14b8a6 50%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 4px 8px rgba(10, 186, 181, 0.3);
        letter-spacing: -0.02em;
        filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3));
        font-family: 'Poppins', sans-serif !important;
    }
    
    .sub-header {
        text-align: center;
        color: #2d3748;
        margin-bottom: 2rem;
        font-size: 1.3rem;
        font-weight: 500;
        background: rgba(10, 186, 181, 0.1);
        padding: 1.2rem 2rem;
        border-radius: 15px;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(10, 186, 181, 0.2);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        font-family: 'Poppins', sans-serif !important;
    }
    
    .upload-section {
        background: linear-gradient(135deg, rgba(10, 186, 181, 0.1) 0%, rgba(255, 255, 255, 0.8) 100%);
        padding: 2rem;
        border-radius: 16px;
        border: 2px dashed rgba(10, 186, 181, 0.5);
        margin: 1.5rem 0;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        color: #2d3748;
    }
    
    .upload-section h3 {
        color: #2d3748 !important;
        font-weight: 700 !important;
        font-size: 1.4rem !important;
        margin-bottom: 0.5rem !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    .upload-section p {
        color: #4a5568 !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    .upload-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(10, 186, 181, 0.05), transparent);
        transform: rotate(45deg);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .user-message {
        background: linear-gradient(135deg, #0ABAB5 0%, #06a19b 100%);
        color: white;
        padding: 1.4rem 1.8rem;
        border-radius: 18px;
        margin: 1rem 0;
        border-left: 4px solid #047481;
        box-shadow: 0 10px 30px rgba(10, 186, 181, 0.25);
        font-weight: 500;
        position: relative;
        overflow: hidden;
        font-size: 1.05rem;
        line-height: 1.6;
        font-family: 'Poppins', sans-serif !important;
    }
    
    .user-message::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.15), transparent);
        transform: translateX(-100%);
        transition: transform 0.6s;
    }
    
    .user-message:hover::before {
        transform: translateX(100%);
    }
    
    .user-message strong {
        font-weight: 700 !important;
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.95);
    }
    
    .bot-message {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(245, 245, 245, 0.9) 100%);
        color: #2d3748;
        padding: 1.4rem 1.8rem;
        border-radius: 18px;
        margin: 1rem 0;
        border-left: 4px solid #0ABAB5;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        font-weight: 500;
        position: relative;
        overflow: hidden;
        font-size: 1.05rem;
        line-height: 1.7;
        font-family: 'Poppins', sans-serif !important;
    }
    
    .bot-message::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent, rgba(10, 186, 181, 0.03), transparent);
        transform: translateX(-100%);
        transition: transform 0.6s;
    }
    
    .bot-message:hover::before {
        transform: translateX(100%);
    }
    
    .bot-message strong {
        font-weight: 700 !important;
        font-size: 1.1rem;
        color: #0ABAB5;
    }
    
    /* Professional Sidebar */
    .css-1d391kg, .css-18e3th9, section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.95) 0%, rgba(240, 248, 255, 0.9) 50%, rgba(236, 253, 245, 0.9) 100%);
        backdrop-filter: blur(20px);
        border-radius: 15px;
        border: 1px solid rgba(10, 186, 181, 0.2);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        color: #2d3748;
    }
    
    .css-1d391kg > div {
        background: transparent !important;
    }
    
    .feature-box {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(10, 186, 181, 0.05) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        margin: 0.8rem 0;
        text-align: center;
        border: 1px solid rgba(10, 186, 181, 0.2);
        backdrop-filter: blur(15px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        color: #2d3748;
        font-weight: 600;
        font-size: 1rem;
        font-family: 'Poppins', sans-serif !important;
    }
    
    .feature-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(10, 186, 181, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .feature-box:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 15px 35px rgba(10, 186, 181, 0.2);
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(10, 186, 181, 0.1) 100%);
        border-color: rgba(10, 186, 181, 0.4);
        color: #1a365d;
    }
    
    .feature-box:hover::before {
        left: 100%;
    }
    
    /* Professional Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #0ABAB5 0%, #06a19b 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.9rem 2.2rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 8px 25px rgba(10, 186, 181, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        letter-spacing: 0.5px;
        font-family: 'Poppins', sans-serif !important;
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
        box-shadow: 0 12px 35px rgba(10, 186, 181, 0.4);
        background: linear-gradient(135deg, #06a19b 0%, #047481 100%);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:active {
        transform: translateY(0px);
        box-shadow: 0 6px 20px rgba(10, 186, 181, 0.3);
    }
    
    /* Enhanced Input Field Styling */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid rgba(10, 186, 181, 0.3);
        padding: 1rem 1.4rem;
        background: rgba(255, 255, 255, 0.9);
        color: #2d3748;
        font-size: 1rem;
        font-weight: 500;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        font-family: 'Poppins', sans-serif !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(45, 55, 72, 0.6);
        font-style: italic;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #0ABAB5;
        box-shadow: 0 0 20px rgba(10, 186, 181, 0.2);
        background: rgba(255, 255, 255, 0.98);
        outline: none;
    }
    
    /* Professional Section Headers */
    h3, .stMarkdown h3 {
        color: #2d3748 !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(10, 186, 181, 0.08) 100%) !important;
        padding: 1rem 1.8rem !important;
        border-radius: 12px !important;
        backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(10, 186, 181, 0.2) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08) !important;
        margin-bottom: 1.5rem !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Regular text styling */
    .stMarkdown p, .stMarkdown div, .stText {
        color: #2d3748 !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Radio button container */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.8);
        padding: 1.4rem;
        border-radius: 12px;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(10, 186, 181, 0.2);
        box-shadow: 0 6px 20px rgba(10, 186, 181, 0.08);
        color: #2d3748;
    }
    
    .stRadio label {
        color: #2d3748 !important;
        font-weight: 500 !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Metrics styling */
    .css-1r6slb0 {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(10, 186, 181, 0.05) 100%);
        border-radius: 12px;
        padding: 1.4rem;
        box-shadow: 0 8px 25px rgba(10, 186, 181, 0.1);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(10, 186, 181, 0.2);
        color: #2d3748;
    }
    
    /* Enhanced File uploader styling */
    .css-1cpxqw2, 
    section[data-testid="stFileUploader"],
    .css-1cpxqw2 > div,
    div[data-testid="stFileUploader"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(10, 186, 181, 0.05) 100%) !important;
        border-radius: 16px !important;
        border: 2px dashed rgba(10, 186, 181, 0.4) !important;
        padding: 2rem !important;
        box-shadow: 0 10px 30px rgba(10, 186, 181, 0.1) !important;
        transition: all 0.3s ease !important;
        position: relative !important;
    }
    
    div[data-testid="stFileUploader"]:hover {
        border-color: rgba(10, 186, 181, 0.6) !important;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(10, 186, 181, 0.08) 100%) !important;
        transform: translateY(-2px) !important;
    }
    
    /* File uploader text styling */
    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] p,
    div[data-testid="stFileUploader"] small {
        color: #2d3748 !important;
        font-weight: 600 !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Browse files button styling */
    div[data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #0ABAB5 0%, #06a19b 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 1rem 2rem !important;
        font-weight: 600 !important;
        box-shadow: 0 8px 25px rgba(10, 186, 181, 0.3) !important;
        transition: all 0.3s ease !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 0.95rem !important;
    }
    
    div[data-testid="stFileUploader"] button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 30px rgba(10, 186, 181, 0.4) !important;
        background: linear-gradient(135deg, #06a19b 0%, #047481 100%) !important;
    }
    
    /* File upload icon styling */
    div[data-testid="stFileUploader"] svg {
        color: #0ABAB5 !important;
        filter: drop-shadow(0 2px 4px rgba(10, 186, 181, 0.3)) !important;
    }
    
    /* Sidebar content styling */
    .css-18e3th9 {
        background: transparent !important;
        padding: 1.2rem;
        border-radius: 15px;
    }
    
    /* Sidebar text styling */
    section[data-testid="stSidebar"] h3 {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(10, 186, 181, 0.08) 100%) !important;
        padding: 1rem 1.4rem !important;
        border-radius: 12px !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(10, 186, 181, 0.2) !important;
        color: #2d3748 !important;
        box-shadow: 0 6px 20px rgba(10, 186, 181, 0.08) !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar metrics styling */
    section[data-testid="stSidebar"] .css-1r6slb0 {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(10, 186, 181, 0.05) 100%);
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 6px 20px rgba(10, 186, 181, 0.08);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(10, 186, 181, 0.2);
        color: #2d3748;
    }
    
    /* Sidebar info box styling */
    section[data-testid="stSidebar"] .stAlert {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(10, 186, 181, 0.08) 100%);
        color: #2d3748;
        border-radius: 12px;
        border: 1px solid rgba(10, 186, 181, 0.2);
        backdrop-filter: blur(15px);
        box-shadow: 0 6px 20px rgba(10, 186, 181, 0.08);
    }
    
    /* Sidebar markdown text */
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] div {
        color: #2d3748 !important;
        font-weight: 500 !important;
        font-family: 'Poppins', sans-serif !important;
    }

    /* Enhanced Success, Error, and Warning messages */
    .stSuccess {
        background: linear-gradient(135deg, #0ABAB5 0%, #06a19b 100%);
        color: white !important;
        border-radius: 12px;
        border: 1px solid rgba(10, 186, 181, 0.3);
        box-shadow: 0 8px 25px rgba(10, 186, 181, 0.2);
        font-family: 'Poppins', sans-serif !important;
        font-weight: 500 !important;
    }
    
    .stError {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white !important;
        border-radius: 12px;
        border: 1px solid rgba(239, 68, 68, 0.3);
        box-shadow: 0 8px 25px rgba(239, 68, 68, 0.2);
        font-family: 'Poppins', sans-serif !important;
        font-weight: 500 !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white !important;
        border-radius: 12px;
        border: 1px solid rgba(245, 158, 11, 0.3);
        box-shadow: 0 8px 25px rgba(245, 158, 11, 0.2);
        font-family: 'Poppins', sans-serif !important;
        font-weight: 500 !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(10, 186, 181, 0.1);
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #0ABAB5 0%, #06a19b 100%);
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #06a19b 0%, #047481 100%);
    }
    
    /* Loading spinner animation */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Selection styling */
    ::selection {
        background: rgba(10, 186, 181, 0.2);
        color: #2d3748;
    }
    
    ::-moz-selection {
        background: rgba(10, 186, 181, 0.2);
        color: #2d3748;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9);
        border: 2px solid rgba(10, 186, 181, 0.3);
        border-radius: 12px;
        color: #2d3748;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 500;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #0ABAB5;
        box-shadow: 0 0 20px rgba(10, 186, 181, 0.2);
    }
    
    /* Dropdown options */
    .stSelectbox div[data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.95);
        color: #2d3748;
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #0ABAB5 0%, #06a19b 100%);
    }
    
    /* Metric styling */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(10, 186, 181, 0.05) 100%);
        border: 1px solid rgba(10, 186, 181, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        backdrop-filter: blur(10px);
        color: #2d3748;
    }
    
    div[data-testid="metric-container"] > label {
        color: #2d3748 !important;
        font-weight: 600 !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    div[data-testid="metric-container"] > div {
        color: #0ABAB5 !important;
        font-weight: 700 !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        border: 1px solid rgba(10, 186, 181, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #2d3748;
        font-weight: 600;
        font-family: 'Poppins', sans-serif !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0ABAB5 0%, #06a19b 100%);
        color: white !important;
    }
    
    /* Download button special styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(10, 186, 181, 0.1) 100%);
        color: #2d3748;
        border: 2px solid rgba(10, 186, 181, 0.3);
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        font-family: 'Poppins', sans-serif !important;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #0ABAB5 0%, #06a19b 100%);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(10, 186, 181, 0.3);
    }
    
    /* Footer enhancement */
    .footer-container {
        background: linear-gradient(135deg, #0ABAB5 0%, #06a19b 50%, #047481 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        margin-top: 2rem;
        box-shadow: 0 12px 40px rgba(10, 186, 181, 0.3);
        position: relative;
        overflow: hidden;
        text-align: center;
    }
    
    .footer-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 30% 70%, rgba(255, 255, 255, 0.1) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .footer-container h3 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
        background: none !important;
        color: white !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    .footer-container p {
        position: relative;
        z-index: 1;
        margin: 1rem 0;
        font-size: 1.1rem;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.95) !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Responsive design improvements */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .sub-header {
            font-size: 1.1rem;
            padding: 1rem 1.5rem;
        }
        
        .upload-section {
            padding: 1.5rem;
        }
        
        .feature-box {
            padding: 1.2rem;
        }
        
        .main .block-container {
            padding: 1.5rem;
            margin: 0.5rem;
        }
        
        .user-message, .bot-message {
            padding: 1.2rem 1.4rem;
            font-size: 1rem;
        }
        
        .footer-container h3 {
            font-size: 1.8rem;
        }
    }
    
    /* Additional professional touches */
    .stMarkdown {
        color: #2d3748 !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 700 !important;
        color: #2d3748 !important;
    }
    
    /* Code block styling */
    .stCode {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(10, 186, 181, 0.2);
        border-radius: 10px;
        color: #2d3748;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 12px;
        backdrop-filter: blur(10px);
        font-family: 'Poppins', sans-serif !important;
        font-weight: 500;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(10, 186, 181, 0.05) 100%);
        color: #2d3748;
        border-radius: 12px;
        border: 1px solid rgba(10, 186, 181, 0.2);
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600;
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.6);
        border: 1px solid rgba(10, 186, 181, 0.15);
        border-radius: 0 0 12px 12px;
        color: #2d3748;
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Enhanced spinner styling */
    .stSpinner > div {
        border-top-color: #0ABAB5 !important;
    }
    
    /* Conversation history section styling */
    .conversation-title {
        color: #2d3748 !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(10, 186, 181, 0.08) 100%) !important;
        padding: 1.2rem 2rem !important;
        border-radius: 15px !important;
        backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(10, 186, 181, 0.2) !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08) !important;
        margin-bottom: 1.5rem !important;
        font-family: 'Poppins', sans-serif !important;
        text-align: center;
    }
    
    /* Question ask section styling */
    .question-section {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(10, 186, 181, 0.05) 100%);
        padding: 2rem;
        border-radius: 18px;
        border: 1px solid rgba(10, 186, 181, 0.2);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        margin: 1.5rem 0;
        backdrop-filter: blur(15px);
    }
    
    /* Enhanced link styling */
    a {
        color: #0ABAB5 !important;
        text-decoration: none !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    a:hover {
        color: #06a19b !important;
        text-shadow: 0 2px 4px rgba(10, 186, 181, 0.3) !important;
    }
    
    /* Label styling improvements */
    label {
        color: #2d3748 !important;
        font-weight: 600 !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 1rem !important;
    }
    
    /* Info box styling */
    .stInfo {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(147, 197, 253, 0.05) 100%);
        color: #1e40af !important;
        border-radius: 12px;
        border: 1px solid rgba(59, 130, 246, 0.2);
        backdrop-filter: blur(15px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.08);
        font-family: 'Poppins', sans-serif !important;
        font-weight: 500 !important;
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
st.markdown('<div class="question-section">', unsafe_allow_html=True)
st.markdown("### 💬 Ask Questions About Your Document")

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

st.markdown('</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="conversation-title">💭 Conversation History</div>', unsafe_allow_html=True)
    
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
            <strong>AI Assistant:</strong> {answer}
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
        if st.button("🔊 Listen to Answer", use_container_width=True):
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
        if st.button("💾 Export Chat", use_container_width=True):
            chat_text = "\n".join([f"Q: {chat[0]}\nA: {chat[1]}\n{'-'*50}\n" for chat in st.session_state.chat_history])
            st.download_button(
                "📥 Download Chat History", 
                chat_text, 
                file_name="chat_history.txt",
                mime="text/plain",
                use_container_width=True
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        if st.button("🗑️ Clear All", use_container_width=True):
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
    st.markdown("### ⭐ How was the AI response?")
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        feedback = st.radio(
            "Rate the response quality:",
            ["⭐ Excellent", "👍 Good", "👎 Needs improvement"],
            horizontal=True
        )
    
    with col2:
        if feedback:
            st.success(f"✅ Thank you for the feedback: {feedback}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div class="footer-container">
        <h3>📄 AI PDF Chatbot</h3>
        <p style='font-size: 1.2rem; margin: 1rem 0; font-weight: 600;'>
            <strong>Upload PDF → Ask Questions → Get Intelligent Answers</strong>
        </p>
        <p style='font-size: 1rem; opacity: 0.9; margin: 0; font-weight: 400;'>
            Experience next-generation document analysis with modern AI assistance
        </p>
    </div>
    """, 
    unsafe_allow_html=True)
