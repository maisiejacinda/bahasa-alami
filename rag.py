import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="UAS RAG Bot - Informatika", page_icon="🤖", layout="wide")

# Custom CSS agar UI lebih cantik
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 QA Bot RAG - Sistem Informasi Dokumen")
st.info("Gunakan aplikasi ini untuk bertanya langsung kepada isi dokumen PDF Anda.")

# --- SIDEBAR KONFIGURASI ---
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    # Link untuk ambil API Key Groq gratis
    st.markdown("[Dapatkan Groq API Key Gratis di sini](https://console.groq.com/keys)")
    groq_key = st.text_input("Masukkan Groq API Key (gsk_...)", type="password")
    
    st.divider()
    
    uploaded_file = st.file_uploader("📂 Unggah PDF (Dataset)", type="pdf")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()

    st.divider()
    st.caption("Dibuat untuk Tugas UAS Teknik Informatika")

# --- FUNGSI PEMROSESAN PDF (RAG) ---
def process_pdf(file):
    # Simpan file sementara agar bisa dibaca loader
    with open("temp.pdf", "wb") as f:
        f.write(file.getvalue())
    
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    
    # Chunking: Memecah teks agar relevan bagi AI
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    
    # Embedding: Mengubah teks jadi angka secara GRATIS (tanpa API Key)
    # Model: all-MiniLM-L6-v2 sangat ringan dan cepat
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Simpan ke Vector Store lokal (FAISS)
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

# --- LOGIKA UTAMA APLIKASI ---
if uploaded_file and groq_key:
    # 1. Inisialisasi Vector Database
    if "vector_db" not in st.session_state:
        with st.status("🛠️ Sedang memproses dokumen...", expanded=True) as status:
            st.write("Membaca PDF...")
            st.session_state.vector_db = process_pdf(uploaded_file)
            st.write("Membangun basis data vektor...")
            status.update(label="✅ Dokumen siap dianalisis!", state="complete", expanded=False)

    # 2. Inisialisasi Memori Chat
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            output_key='answer'
        )

    # 3. Inisialisasi Model Chat (Groq - Llama 3)
    try:
        llm = ChatGroq(
            temperature=0.2, 
            groq_api_key=groq_key, 
            model_name="llama-3.1-8b-instant"
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.vector_db.as_retriever(search_kwargs={"k": 3}),
            memory=st.session_state.memory,
            return_source_documents=True
        )

        # 4. Tampilkan Riwayat Chat
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        # 5. Input User & Respon AI
        if prompt := st.chat_input("Tanya sesuatu tentang isi PDF..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Berpikir..."):
                    result = qa_chain.invoke({"question": prompt})
                    response = result["answer"]
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

else:
    # Tampilan awal jika belum upload/isi key
    st.warning("⚠️ Langkah Persiapan:")
    col1, col2 = st.columns(2)
    with col1:
        st.write("1. Masukkan API Key Groq di sidebar kiri.")
    with col2:
        st.write("2. Unggah file PDF yang ingin ditanyakan.")
