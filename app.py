from google import genai
from google.genai import types
import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Sayfa Ayarları
st.set_page_config(page_title="Zemax Chatbot", page_icon="🔭", layout="centered")
st.title("🔭 Zemax Optik Asistanı")

# 2. API Key Güvenlik Kontrolü
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("🔑 Gemini API Key bulunamadı! Lütfen Streamlit ayarlarından Secrets kısmını kontrol edin.")
    st.stop()

# 3. Gemini İstemcisini Başlatma
@st.cache_resource
def get_gemini_client(key):
    return genai.Client(api_key=key)

client = get_gemini_client(api_key)

# 4. Hafızada (RAM) Otomatik Vektör Veritabanı Oluşturma
@st.cache_resource
def initialize_vector_db():
    txt_path = "Zemax_operand.txt"
    
    if not os.path.exists(txt_path):
        st.error(f"📂 Ana belge olan '{txt_path}' GitHub reposunda bulunamadı!")
        return None
        
    try:
        # Belgeyi yükle
        loader = TextLoader(txt_path, encoding="utf-8")
        docs = loader.load()
        
        # Parçalara ayır
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
        chunks = splitter.split_documents(docs)
        
        # Embedding modelini yükle
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # FAISS veritabanını diske kaydetmeden direkt RAM'de oluştur
        local_db = FAISS.from_documents(chunks, embeddings)
        return local_db
    except Exception as e:
        st.error(f"Veritabanı hazırlanırken bir hata oluştu: {str(e)}")
        return None

# Uygulama başlarken otomatik tetiklenir ve cache'lenir (Sadece ilk açılışta 10-15 sn sürer)
db = initialize_vector_db()

# 5. Kullanıcı Arayüzü ve Soru-Cevap
query = st.text_input("Zemax veya optik tasarım hakkında bir soru bakın:", placeholder="Örn: Mercek optimizasyonu nasıl yapılır?")

if query:
    if not db:
        st.error("Veritabanı hazır olmadığı için arama yapılamıyor.")
        st.stop()
        
    with st.spinner("🔍 Dokümanlar taranıyor ve yanıt hazırlanıyor..."):
        try:
            # Kullanıcının sorusuna en yakın 4 metin parçasını hafızadan çekiyoruz
            docs = db.similarity_search(query, k=4)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Sistem talimatları ve prompt yapısı
            system_instruction = (
                "Sen bir optik tasarım ve Zemax uzmanısın. Kullanıcının sorularına, "
                "sana verilen doküman kaynaklarına (Context) dayanarak teknik, net ve profesyonel cevaplar ver. "
                "Eğer cevap dokümanda yoksa, kendi genel optik bilgini kullanarak en doğru yönlendirmeyi yap."
            )
            
            prompt = f"Kullanıcı Sorusu: {query}\n\nDokümandan İlgili Bilgiler (Context):\n{context}"
            
            # Gemini içerik üretimi
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3,
                ),
            )
            
            # Sonucu Ekrana Yazma
            st.success("Cevap:")
            st.write(response.text)
            
            # Kaynak gösterimi
            with st.expander("📚 Yararlanılan Doküman Kesitleri"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**Kesit {i}:**")
                    st.caption(doc.page_content)
                    
        except Exception as e:
            st.error(f"Yanıt üretilirken bir hata oluştu: {str(e)}")