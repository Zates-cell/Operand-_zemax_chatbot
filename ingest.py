from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os

# .env dosyasını yükle
load_dotenv()

# API key'i otomatik çek
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("API key bulunamadı. .env dosyasını kontrol et.")
    exit()

# text dosyasını oku
with open("data/zemax_operands.txt", "r", encoding="utf-8") as f:
    text = f.read()

# text parçalama
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.create_documents([text])

# embedding modeli
embeddings = OpenAIEmbeddings(
    openai_api_key=api_key
)

# vector database oluştur
vectorstore = FAISS.from_documents(docs, embeddings)

# kaydet
vectorstore.save_local("faiss_index")

print("Vector database başarıyla oluşturuldu.")