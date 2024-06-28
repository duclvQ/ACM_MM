import streamlit as st
import random
import time
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

@st.cache_resource
def load_ollama_llamma3():
    llm = ChatOllama(model="llama3", temperature=0.2,  top_p=1,)
    return llm

llm = load_ollama_llamma3()
# keypoints_prompt = ChatPromptTemplate.from_template("Reutn the keypoints of meeting below: {transcripts} ")
# keypoints_chain = keypoints_prompt | llm | StrOutputParser()