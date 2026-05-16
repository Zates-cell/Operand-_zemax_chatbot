from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

# OpenAI key
api_key = os.getenv("OPENAI_API_KEY")

# 📄 TXT dosyan
loader = TextLoader("Zemax_operand.txt", encoding="utf-8")
docs = loader.load()

# chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

# embeddings
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# FAISS index
db = FAISS.from_documents(chunks, embeddings)

db.save_local("faiss_index")

print("FAISS index hazır ✔")