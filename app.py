import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(page_title="Zemax Chatbot", page_icon="🔭")

st.title("🔭 Zemax RAG Chatbot (Gemini)")

# =========================
# 🔑 GEMINI API KEY
# =========================
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key yok")
    st.stop()

genai.configure(api_key=api_key)

model = genai.GenerativeModel("models/gemini-1.5-flash")

# =========================
# 🔥 EMBEDDING MODEL
# =========================
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# =========================
# 🧠 FAISS AUTO FIX (EN ÖNEMLİ KISIM)
# =========================
if os.path.exists("faiss_index"):
    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    st.warning("FAISS index bulunamadı. Otomatik oluşturuluyor...")

    loader = TextLoader("Zemax_operand.txt", encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("faiss_index")

    st.success("FAISS index oluşturuldu ✔")

# =========================
# 💬 USER INPUT
# =========================
query = st.text_input("Sorunu yaz:")

if query:
    with st.spinner("Zemax dokümanları aranıyor..."):

        docs = db.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
You are an expert in Zemax optical design.

Use ONLY the context below.

CONTEXT:
{context}

QUESTION:
{query}
"""

        response = model.generate_content(prompt)

        st.success("Yanıt:")
        st.write(response.text)