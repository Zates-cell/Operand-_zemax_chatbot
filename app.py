import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# yeni chain importları
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

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
        st.error("OPENAI_API_KEY bulunamadı.")
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

retriever = db.as_retriever(search_kwargs={"k": 3})

# LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=api_key
)

prompt = ChatPromptTemplate.from_template("""
Answer the user's question based only on the provided context.

Context:
{context}

Question:
{input}
""")

document_chain = create_stuff_documents_chain(
    llm,
    prompt
)

retrieval_chain = create_retrieval_chain(
    retriever,
    document_chain
)

question = st.text_input("Sorunu yaz:")

if question:
    try:
        response = retrieval_chain.invoke({
            "input": question
        })

        st.write(response["answer"])

    except Exception as e:
        st.error(f"Hata oluştu: {e}")