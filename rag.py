import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ======================
# UI SETUP
# ======================
st.set_page_config(page_title="UAS RAG Bot", layout="wide")
st.title("🤖 QA Bot RAG - Teknik Informatika")

with st.sidebar:
    st.header("Konfigurasi")
    api_key = st.text_input("Masukkan OpenAI API Key", type="password")
    uploaded_file = st.file_uploader("Unggah PDF (Dataset)", type="pdf")


# ======================
# PROCESS PDF
# ======================
def process_pdf(file, key):
    with open("temp.pdf", "wb") as f:
        f.write(file.getvalue())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    # ⚡ OPTIMASI CHUNK (lebih cepat)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80
    ).split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=key)

    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db


# ======================
# MAIN APP
# ======================
if uploaded_file and api_key:

    if "vector_db" not in st.session_state:
        st.session_state.vector_db = process_pdf(uploaded_file, api_key)

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            k=2  # ⚡ batasi memory biar ringan
        )

    # ⚡ retriever dipercepat
    retriever = st.session_state.vector_db.as_retriever(
        search_kwargs={"k": 2}
    )

    # ⚡ model lebih cepat
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=api_key,
        max_tokens=300
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory
    )

    # ======================
    # CHAT HISTORY
    # ======================
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # ======================
    # INPUT USER
    # ======================
    if prompt := st.chat_input("Tanya isi PDF..."):

        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = qa_chain.invoke({"question": prompt})["answer"]
            st.markdown(response)

            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
