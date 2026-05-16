from google import genai
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("API key yok")
    st.stop()

client = genai.Client(api_key=api_key)

st.title("🔭 Zemax Chatbot")

query = st.text_input("Sorunu yaz:")

if query:
    response = client.models.generate_content(
        model="models/gemini-1.5-flash",
        contents=f"You are an optics expert: {query}"
    )

    st.write(response.text)