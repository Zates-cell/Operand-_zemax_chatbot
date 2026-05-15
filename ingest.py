import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY bulunamadı.")

# PDF dosyan
pdf_path = "zemax_document.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"{len(documents)} sayfa yüklendi.")

# text split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

texts = text_splitter.split_documents(documents)

print(f"{len(texts)} chunk oluşturuldu.")

# embedding
embeddings = OpenAIEmbeddings()

# faiss oluştur
vectorstore = FAISS.from_documents(
    texts,
    embeddings
)

# kaydet
vectorstore.save_local("faiss_index")

print("FAISS index başarıyla oluşturuldu.")