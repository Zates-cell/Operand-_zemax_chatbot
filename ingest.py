from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# text dosyasını oku
with open("data/zemax_operands.txt", "r", encoding="utf-8") as f:
    text = f.read()

# parçalama
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.create_documents([text])

# embedding
embeddings = OpenAIEmbeddings(
    openai_api_key="YOUR_OPENAI_API_KEY"
)

# vector db oluştur
vectorstore = FAISS.from_documents(docs, embeddings)

# kaydet
vectorstore.save_local("faiss_index")

print("Vector database oluşturuldu.")