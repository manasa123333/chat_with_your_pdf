from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
# Install Poppler and Tesseract in the runtime environment
os.system("apt-get update && apt-get install -y poppler-utils tesseract-ocr")

secret = os.getenv('Groq_api')

working_dir = os.path.dirname(os.path.abspath(__file__))

def load_documents(file_path):
    # Specify poppler_path and tesseract_path to ensure compatibility
    loader = UnstructuredPDFLoader(
        file_path, 
        poppler_path="/usr/bin", 
        tesseract_path="/usr/bin/tesseract"
    )
    documents = loader.load()
    return documents

def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(
        separator="/n", 
        chunk_size=1000,
        chunk_overlap=200
    )
    doc_chunks = text_splitter.split_documents(documents)
    vectorstores = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstores

def create_chain(vectorstores):
    llm = ChatGroq(
        api_key=secret,
        model="llama-3.1-8b-instant",
        temperature=0
    )
    retriever = vectorstores.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )
    return chain

# Streamlit page configuration
st.set_page_config(
    page_title="Chat with your documents",
    page_icon="📑",
    layout="centered"
)

st.title("📝Chat With your docs 😎")

# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader(label="Upload your PDF")

if uploaded_file:
    file_path = f"{working_dir}/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if "vectorstores" not in st.session_state:
        st.session_state.vectorstores = setup_vectorstore(load_documents(file_path))

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = create_chain(st.session_state.vectorstores)

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input handling
user_input = st.chat_input("Ask any questions relevant to uploaded pdf")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.session_state.conversation_chain({"question": user_input})
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
