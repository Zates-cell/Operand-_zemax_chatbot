import streamlit as st
import os
from dotenv import load_dotenv

# yeni importlar
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# env yükleme
load_dotenv()

st.set_page_config(
    page_title="Zemax Chatbot",
    page_icon="🔭"
)

st.title("🔭 Zemax Chatbot")

# API KEY alma
api_key = os.getenv("OPENAI_API_KEY")

# Streamlit secrets fallback
if not api_key:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except:
        st.error("OPENAI_API_KEY bulunamadı.")
        st.stop()

# embedding
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

retriever = db.as_retriever(search_kwargs={"k": 3})

# LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=api_key
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

question = st.text_input("Sorunuzu yaz:")

if question:
    try:
        answer = qa_chain.run(question)
        st.write(answer)
    except Exception as e:
        st.error(f"Sorgu hatası: {e}")