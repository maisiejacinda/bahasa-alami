import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Ganti bagian impor yang lama dengan ini
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# UI Setup
st.set_page_config(page_title="UAS RAG Bot", layout="wide")
st.title("🤖 QA Bot RAG - Teknik Informatika")

with st.sidebar:
    st.header("Konfigurasi")
    api_key = st.text_input("Masukkan OpenAI API Key", type="password")
    uploaded_file = st.file_uploader("Unggah PDF (Dataset)", type="pdf")

def process_pdf(file, key):
    with open("temp.pdf", "wb") as f:
        f.write(file.getvalue())
    loader = PyPDFLoader("temp.pdf")
    # Chunking dokumen untuk memproses data
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(loader.load())
    # Membuat vector database menggunakan FAISS
    vector_db = FAISS.from_documents(chunks, OpenAIEmbeddings(openai_api_key=key))
    return vector_db

if uploaded_file and api_key:
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = process_pdf(uploaded_file, api_key)

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Inisialisasi chain untuk tanya jawab dokumen
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key),
        retriever=st.session_state.vector_db.as_retriever(),
        memory=st.session_state.memory
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("Tanya isi PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            response = qa_chain.invoke({"question": prompt})["answer"]
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
