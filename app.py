import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

st.set_page_config(page_title="Zemax Chatbot", page_icon="🔭")

st.title("🔭 Zemax RAG Chatbot (Gemini)")

# 🔑 GEMINI KEY
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key yok")
    st.stop()

genai.configure(api_key=api_key)

model = genai.GenerativeModel("models/gemini-1.5-flash")

# 🔥 embedding (local)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# FAISS yükle
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

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