import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

# .env yükle (lokal kullanım için)
load_dotenv()

st.set_page_config(
    page_title="Zemax Chatbot",
    page_icon="🔭"
)

st.title("🔭 Zemax Chatbot (Gemini Free)")
st.write("Zemax optik tasarımı hakkında soru sorabilirsiniz.")

# API key kontrolü
api_key = None

# Önce Streamlit secrets kontrol et
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]

# Secrets yoksa local .env kontrol et
if not api_key:
    api_key = os.getenv("GEMINI_API_KEY")

# Hala yoksa hata ver
if not api_key:
    st.error("Gemini API key bulunamadı. Streamlit Secrets veya .env kontrol et.")
    st.stop()

# Gemini configure
genai.configure(api_key=api_key)

# Model seç
model = genai.GenerativeModel("gemini-1.5-flash")

# Kullanıcı input
query = st.text_input("Sorunu yaz:")

if query:
    with st.spinner("Yanıt hazırlanıyor..."):
        try:
            prompt = f"""
You are an expert assistant specialized in Zemax, optics, ray tracing, and optical design.

User question:
{query}
"""

            response = model.generate_content(prompt)

            st.success("Yanıt:")
            st.write(response.text)

        except Exception as e:
            st.error("Gemini API hatası oluştu.")
            st.write(f"Detay: {e}")