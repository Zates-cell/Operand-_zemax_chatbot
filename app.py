import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

st.set_page_config(
    page_title="Zemax Chatbot",
    page_icon="🔭"
)

st.title("🔭 Zemax Chatbot (Gemini Free)")
st.write("Zemax optik tasarımı hakkında soru sorabilirsiniz.")

# API KEY
api_key = None

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key bulunamadı.")
    st.stop()

# Gemini config
genai.configure(api_key=api_key)

# Güncel model adı
model = genai.GenerativeModel("gemini-1.5-flash-latest")

query = st.text_input("Sorunu yaz:")

if query:
    with st.spinner("Yanıt hazırlanıyor..."):
        try:
            prompt = f"""
You are an expert in Zemax, optics, ray tracing and optical design.

User question:
{query}
"""

            response = model.generate_content(prompt)

            st.success("Yanıt:")
            st.write(response.text)

        except Exception as e:
            st.error("Gemini API hatası oluştu.")
            st.write(f"Detay: {e}")