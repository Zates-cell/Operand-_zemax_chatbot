import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

# .env yükle (lokal kullanım için)
load_dotenv()

st.set_page_config(
    page_title="Zemax Chatbot",
    page_icon="🔭"
)

st.title("🔭 Zemax Chatbot (Stable)")
st.write("Zemax optik tasarımı hakkında soru sorabilirsiniz.")

# API key kontrolü
api_key = None

# Önce Streamlit secrets kontrol et
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]

# Secrets yoksa local .env kontrol et
if not api_key:
    api_key = os.getenv("OPENAI_API_KEY")

# Hala yoksa hata ver
if not api_key:
    st.error("OpenAI API key bulunamadı. Streamlit Secrets veya .env kontrol et.")
    st.stop()

# OpenAI client
client = OpenAI(api_key=api_key)

# Kullanıcı input
query = st.text_input("Sorunu yaz:")

if query:
    with st.spinner("Yanıt hazırlanıyor..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert assistant specialized in Zemax, optics, ray tracing, and optical design."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            )

            st.success("Yanıt:")
            st.write(response.choices[0].message.content)

        except Exception as e:
            st.error(
                "OpenAI API quota/rate limit hatası oluştu. Billing veya usage limitini kontrol et."
            )
            st.write(f"Detay: {e}")