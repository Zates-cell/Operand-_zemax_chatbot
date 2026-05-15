import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

st.title("🔭 Zemax Chatbot (Stable Version)")

api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not api_key:
    st.error("API key bulunamadı")
    st.stop()

client = OpenAI(api_key=api_key)

embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# FAISS load
try:
    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    st.error(f"Index yüklenemedi: {e}")
    st.stop()

query = st.text_input("Sorunu yaz")

if query:
    docs = db.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a technical assistant.

Use only this context:

{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    st.write(response.choices[0].message.content)