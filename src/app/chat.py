import streamlit as st
import random
import time

from model import llm
from upload_button import upload_text_file
# Streamed response emulator
def response_generator(input_text):
    for chunks in llm.stream(input_text):
        yield chunks

st.title("Simple chat")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # warm up model
    print("Warming up model...")
    llm.invoke('start..')
    

# set the upload state      
upload_state = upload_text_file()
print("upload_state:",upload_state)
if upload_state:
    # remove the previous messages
    st.session_state.messages.clear()
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})