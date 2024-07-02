import streamlit as st
import random
import time
import os
from model import llm, keypoints_generator,meeting_minutes_generator,\
    chain_of_verification, verify_question_generator, extract_keypoints
from upload_button import upload_text_file
# Streamed response emulator


st.title("Simple chat")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # warm up model
    print("Warming up model...")
    llm.invoke('start..')
    
if "upload_name" not in st.session_state:
    st.session_state.upload_name = None
def load_raw_document():
    upload_state, raw_document = upload_text_file(uploaded_file)
    print("upload_state:",upload_state)
    if upload_state:
        # remove the previous messages
        st.session_state.messages.clear()
    return upload_state,raw_document
# set the upload state  
uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "mp3", "wav"])
raw_document = None
if uploaded_file != None: 
    
    if uploaded_file.name != st.session_state.upload_name:
            upload_state, raw_document = load_raw_document()
            st.session_state.upload_name = uploaded_file.name
    else:
        docs_dir = os.path.join("uploads", uploaded_file.name.split(".")[0] + ".txt")
        with open(docs_dir, "rb") as f:
            raw_document = f.read()


    

    
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
        response = st.write_stream(meeting_minutes_generator(raw_document))
        # response = st.write_stream(keypoints_generator(raw_document))
        # kps = extract_keypoints(response)

        # for kp in kps:
        #     print("kp:",kp)
        #     response2 = st.write_stream(verify_question_generator(kp))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})