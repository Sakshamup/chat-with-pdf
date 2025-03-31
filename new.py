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
import speech_recognition as sr
from gtts import gTTS
import tempfile
import re

headers={
    "authorization":st.secrets["GOOGLE_API_KEY"],
    "content-type":"application/json"
}

# Language mapping for gTTS
LANGUAGE_MAPPING = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
}

st.set_page_config(page_title="AI PDF Chatbot", layout="wide")
st.title("📄 AI PDF Chatbot with Enhanced Features")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    language = st.selectbox("Select Language", list(LANGUAGE_MAPPING.keys()))

# Upload PDF
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

@st.cache_data
def extract_text_from_pdfs(files):
    text = ""
    for file in files:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n\n"
    return text

def preprocess_text(text):
    # Remove extra whitespace and normalize text
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_data
def create_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    text_chunks = text_splitter.split_text(text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question based on the provided context. If the answer is not in the context, say "Answer is not available in the context."
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def process_user_message(user_input):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_input)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_input}, return_only_outputs=True)
    return response["output_text"]

# Chat Interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_files:
    raw_text = extract_text_from_pdfs(uploaded_files)
    raw_text = preprocess_text(raw_text)
    create_vector_store(raw_text)
    st.success("PDFs processed successfully! Ask any question.")

user_input = st.text_input("Ask a question about the document")
if st.button("Get Answer") and user_input:
    response = process_user_message(user_input)
    st.session_state.chat_history.append((user_input, response))
    st.write(f"**Bot:** {response}")

for chat in st.session_state.chat_history:
    st.write(f"**You:** {chat[0]}")
    st.write(f"**Bot:** {chat[1]}")

# Speech-to-text
if st.button("🎙️ Speak Question"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
    try:
        spoken_text = recognizer.recognize_google(audio, language=LANGUAGE_MAPPING[language])
        st.text_input("You said:", spoken_text)
        response = process_user_message(spoken_text)
        st.session_state.chat_history.append((spoken_text, response))
        st.write(f"**Bot:** {response}")
    except sr.UnknownValueError:
        st.write("Could not understand the audio. Please try again.")
    except sr.RequestError as e:
        st.write(f"Error: {e}")

# Text-to-Speech
if st.button("🔊 Read Answer"):
    if st.session_state.chat_history:
        tts = gTTS(text=st.session_state.chat_history[-1][1], lang=LANGUAGE_MAPPING[language])
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        st.audio(temp_file.name, format="audio/mp3")
    else:
        st.warning("No response available to read.")

# Export Chat History
if st.button("Export Chat History"):
    if st.session_state.chat_history:
        chat_text = "\n".join([f"You: {chat[0]}\nBot: {chat[1]}\n" for chat in st.session_state.chat_history])
        st.download_button("Download Chat History", chat_text, file_name="chat_history.txt")
    else:
        st.warning("No chat history available to export.")

# Feedback Mechanism
feedback = st.radio("Rate the bot's response:", ["👍 Good", "👎 Bad"])
if feedback:
    st.write(f"Thank you for your feedback: {feedback}")
