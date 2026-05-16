from google import genai
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("API key yok")
    st.stop()

client = genai.Client(api_key=api_key)

st.title("🔭 Zemax Chatbot")

query = st.text_input("Sorunu yaz:")

if query:
    with st.spinner("Yanıt hazırlanıyor..."):
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=f"You are an optics expert. Question: {query}"
        )

        st.write(response.text)