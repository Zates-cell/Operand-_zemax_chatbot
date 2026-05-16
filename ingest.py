from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def main():
    pdf_path = "zemax_document.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Hata: '{pdf_path}' dosyası bulunamadı! Lütfen proje klasörüne ekleyin.")
        return

    print("PDF yükleniyor...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Optik dokümanları formül ve yapısal veri içerdiğinden 
    # chunk_size ve overlap oranını optimize ettik
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)
    print(f"Doküman {len(chunks)} parçaya ayrıldı.")

    print("Embeddings modeli yükleniyor (Lokal)...")
    # Ücretsiz, hızlı ve API key gerektirmeyen endüstri standardı lokal model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("FAISS veritabanı oluşturuluyor...")
    db = FAISS.from_documents(chunks, embeddings)
    
    # Yerel klasör olarak kaydet
    db.save_local("faiss_index")
    print("İşlem başarıyla tamamlandı! 'faiss_index' klasörü oluşturuldu.")

if __name__ == "__main__":
    main()