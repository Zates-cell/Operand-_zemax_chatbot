import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import HumanMessage

load_dotenv()

st.set_page_config(
    page_title="Zemax Chatbot",
    page_icon="🔭"
)

st.title("🔭 Zemax Documentation Chatbot")

# API key alma
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except:
        st.error("OPENAI API key bulunamadı.")
        st.stop()

# embeddings
embeddings = OpenAIEmbeddings(
    openai_api_key=api_key
)

# FAISS yükleme
try:
    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    st.error(f"FAISS yüklenemedi: {e}")
    st.stop()

# model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=api_key
)

question = st.text_input("Sorunuzu yazın")

if question:
    try:
        # benzer dokümanları bul
        docs = db.similarity_search(
            question,
            k=3
        )

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
You are a Zemax documentation assistant.

Use ONLY the context below to answer.

Context:
{context}

Question:
{question}
"""

        response = llm.invoke(
            [HumanMessage(content=prompt)]
        )

        st.write(response.content)

    except Exception as e:
        st.error(f"Hata oluştu: {e}")