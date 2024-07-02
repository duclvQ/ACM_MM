import streamlit as st
import random
import time
import os
from model import   llm, chat_history,\
                    verify_question_generator, \
                    extract_keypoints, \
                    classify_user_need, \
                    base_generator, \
                    compress_transcript, \
                    generate_first_meeting_minute, \
                    extract_fact, \
                    verify_question_generator, \
                    answer_question, \
                    generate_resvised_meeting_minutes
                    
from upload_button import upload_text_file
from langchain_core.messages import HumanMessage, AIMessage
# Streamed response emulator
from prompt_analysis import MeetingSummaryEvaluator
from utils import convert_str_to_json
st.title("Simple chat")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # warm up model
    print("Warming up model...")
    llm.invoke('start..')

if "first_run" not in st.session_state:
    st.session_state.first_run = True
    
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
uploaded_file = st.file_uploader("Choose a file", type=["txt", "mp3", "wav"])
raw_document = None
if uploaded_file != None: 
    
    if uploaded_file.name != st.session_state.upload_name:
            upload_state, raw_document = load_raw_document()
            st.session_state.upload_name = uploaded_file.name
            st.session_state.first_run = True
    else:
        docs_dir = os.path.join("uploads", uploaded_file.name.split(".")[0] + ".txt")
        with open(docs_dir, "rb") as f:
            raw_document = f.read()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
# generate the first meeting minute
if st.session_state.first_run:
    st.write("Please upload a transcript or audio file to start.")
    if raw_document:
        st.session_state.first_run = False
        st.write(':blue[Compress the transcript...]')
        compressed_transcript, rate = compress_transcript(raw_document)
        st.write(':blue[Compression rate:]', rate)
        st.write(':blue[Generate a meeting minute from the compressed transcript...]')
        st.session_state.messages.append({"role": "user", "content": "write a meeting minutes from the transcript."})
        chat_history.append(f"HUMAN: write a meeting minutes from the transcript.\n")
        first_meeting_minutes = st.write_stream(generate_first_meeting_minute(compressed_transcript))
        
        st.write(':blue[Here are the information that need to be rechecked.]')
        verify_questions = st.write_stream(verify_question_generator(first_meeting_minutes))
        # convert to dict
        verify_questions = convert_str_to_json(verify_questions)
        answers = []
        for key in list(verify_questions.keys()):
            for question in verify_questions[key]:
                st.write(f":blue[Question: {question}]")
                answer = st.write_stream(answer_question(question, context = raw_document))
                q_and_a = {"question": question, "answer": answer}
                answers.append(q_and_a)
                st.session_state.messages.append({"role": "assistant", "content": f"Question: {question}\nAnswer: {answer}"})
        # join the answers
        answers = "\n".join([f"Question: {qa['question']}\nAnswer: {qa['answer']}" for qa in answers])
        st.write(':blue[Now, I will generate the revised meeting minutes based on the answers.]')
        revised_meeting_minutes = st.write_stream(generate_resvised_meeting_minutes(raw_document,first_meeting_minutes, answers))
        st.session_state.messages.append({"role": "assistant", "content": revised_meeting_minutes})
        
                
        st.session_state.messages.append({"role": "assistant", "content": revised_meeting_minutes})
        chat_history.append(f"AI: {revised_meeting_minutes}\n")
        st.write(':blue[Do you want me to change anything in the meeting minute?\n \
                    Or You can ask me other questions about this meeting minutes.]')



# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # st.write(f":blue[{chat_history}]")
        # st.write(":blue[classifying user's input...]")
        # classify the user input
        # user_need = classify_user_need(prompt, chat_history)
        # st.write(f":blue[User need is: {user_need}]")
        # Generate the first response
        
        response = st.write_stream(base_generator(input_text=prompt, chat_history=chat_history, context = raw_document))
        
        # Add assistant response to chat history
        chat_history.append(f"HUMAN: {prompt}\n, AI: {response}\n")
    
    st.session_state.messages.append({"role": "assistant", "content": response})