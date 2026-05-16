from google import genai
from google.genai import types
import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Sayfa Ayarları
st.set_page_config(page_title="Zemax Chatbot", page_icon="🔭", layout="centered")
st.title("🔭 Zemax Optik Asistanı")

# 2. API Key Güvenlik Kontrolü
# Streamlit Cloud'daki "Secrets" alanından veya yerel .env dosyasından okur
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("🔑 Gemini API Key bulunamadı! Lütfen Streamlit ayarlarından Secrets kısmını kontrol edin.")
    st.stop()

# 3. Gemini İstemcisini Başlatma (Yeni google-genai SDK formatı)
@st.cache_resource
def get_gemini_client(key):
    return genai.Client(api_key=key)

client = get_gemini_client(api_key)

# 4. Vektör Veritabanını Yükleme (Cache'leyerek performansı artırıyoruz)
@st.cache_resource
def load_vector_db():
    if not os.path.exists("faiss_index"):
        st.error("📂 'faiss_index' klasörü bulunamadı. Lütfen önce ingest.py dosyasını çalıştırın!")
        return None
    
    # ingest.py ile aynı embedding modelini kullanmak zorundayız
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

db = load_vector_db()

# 5. Kullanıcı Etkileşimi
query = st.text_input("Zemax veya optik tasarım hakkında bir soru sorun:", placeholder="Örn: Huygens mercek tasarımı nasıl optimize edilir?")

if query and db:
    with st.spinner("🔍 Dokümanlar taranıyor ve yanıt hazırlanıyor..."):
        try:
            # Kullanıcının sorusuna en yakın 4 metin parçasını getiriyoruz
            docs = db.similarity_search(query, k=4)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Gemini için System Prompt ve Context kurgusu
            system_instruction = (
                "Sen bir optik tasarım ve Zemax uzmanısın. Kullanıcının sorularına, "
                "sana verilen doküman kaynaklarına (Context) dayanarak teknik, net ve profesyonel cevaplar ver. "
                "Eğer cevap dokümanda yoksa, kendi genel optik bilgini kullanarak en doğru yönlendirmeyi yap."
            )
            
            prompt = f"Kullanıcı Sorusu: {query}\n\nDokümandan İlgili Bilgiler (Context):\n{context}"
            
            # Yeni SDK'ya uygun içerik üretme çağrısı
            response = client.models.generate_content(
                model="gemini-1.5-flash", # Hata devam ederse alternatif: "gemini-2.5-flash"
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3, # Teknik doğruluk için yaratıcılığı düşürdük
                ),
            )
            
            # Sonucu Ekrana Yazdırma
            st.success("Cevap:")
            st.write(response.text)
            
            # İsteğe bağlı: Kaynak gösterimi
            with st.expander("📚 Yararlanılan Doküman Kaynakları"):
                for i, doc in enumerate(docs, 1):
                    source_info = doc.metadata.get('source', 'Bilinmeyen Kaynak')
                    page_info = doc.metadata.get('page', 0) + 1
                    st.markdown(f"**Kaynak {i}:** {source_info} (Sayfa: {page_info})")
                    st.caption(doc.page_content[:200] + "...")
                    
        except Exception as e:
            st.error(f"Sistemsel bir hata oluştu: {str(e)}")
            st.info("Eğer hata kütüphane kaynaklıysa model adını 'gemini-2.5-flash' olarak güncellemeyi deneyebilirsiniz.")