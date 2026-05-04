import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

st.title("Zemax Operand Assistant")
st.write("Zemax optimization operand bulucu chatbot")

# vector db yükle
embeddings = OpenAIEmbeddings(
    openai_api_key="YOUR_OPENAI_API_KEY"
)

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

query = st.text_input("Sorunu yaz:")

if query:
    response = qa_chain.run(query)
    st.write("### Cevap:")
    st.write(response)