import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

st.title("🔭 Zemax Chatbot (Stable)")

api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not api_key:
    st.error("API key yok")
    st.stop()

client = OpenAI(api_key=api_key)

# FAISS yok → hata kaynağı tamamen kaldırıldı
# (şimdilik retrieval yok, sadece LLM)

query = st.text_input("Sorunu yaz")

if query:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"Zemax uzmanı gibi cevap ver: {query}"
            }
        ]
    )

    st.write(response.choices[0].message.content)