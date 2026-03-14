import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
st.title("Groq Chatbot")
if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
user_input = st.chat_input("Ask something...")
if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.write(user_input)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=st.session_state.messages
    )
    bot_reply = response.choices[0].message.content
    st.session_state.messages.append(
        {"role": "assistant", "content": bot_reply}
    )
    with st.chat_message("assistant"):
        st.write(bot_reply)