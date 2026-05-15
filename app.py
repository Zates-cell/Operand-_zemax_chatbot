import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# .env dosyasını yükle
load_dotenv()

# API key buradan otomatik okunuyor
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("API key bulunamadı. .env dosyasını kontrol et.")
    st.stop()

st.title("Zemax Operand Assistant")
st.write("Zemax optimization operand öneri chatbotu")

# Embedding modeli
embeddings = OpenAIEmbeddings(
    openai_api_key=api_key
)

# Vector database yükle
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

# Chat modeli
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=api_key
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

query = st.text_input("Optik tasarım problemini yaz:")

if query:
    response = qa_chain.run(query)

    st.write("### Önerilen Operand:")
    st.write(response)