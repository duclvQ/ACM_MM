import streamlit as st
import random
import time
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters import CharacterTextSplitter


from prompt_template import key_points_template, action_items_template



text_splitter = CharacterTextSplitter(
    separator="(",
    chunk_size=3000,
    chunk_overlap=500,
    length_function=len,
    is_separator_regex=False,
)
@st.cache_resource
def load_ollama_llamma3():
    llm = ChatOllama(model="llama3", 
                    temperature=0.5, 
                    top_p=1,
                    mirostat = 2,
                    mirostat_tau =4,
                    )
    return llm


llm = load_ollama_llamma3()
chat_history = []

def key_points_generator(input_text):
    # convert to string
    input_text = str(input_text)
    docs = text_splitter.create_documents([input_text])
    split_docs = text_splitter.split_documents(docs)
    # Define prompt
    prompt_template = """Write a list of key points of the following meeting conversation in form of a meeting minutes:
    "{text}"
    Key points in bullet points format:"""
    prompt = PromptTemplate.from_template(prompt_template)
    # Define LLM chain
    
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    return stuff_chain.invoke(split_docs)

def classify_user_need(input_text, chat_history):
    prompt_template = """SYSTEM 
    Classify the user need based on the chat conversation below,
    Whether the user need is "Question" or an "Action".
    RETURN your answer in 1 word:
    
    Chat_history
    {chat_history}
    
    HUMAN
    {input_text}
    
    Answer :"""
    prompt = PromptTemplate.from_template(prompt_template)
    classify_chain = prompt | llm | StrOutputParser()
    return classify_chain.invoke({"input_text": input_text, "chat_history": chat_history[-2:]})

def base_generator(input_text, chat_history, context):
    prompt = """
    SYSTEM 
    You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. 
    
    Context: {context} 
    Chat history: {chat_history}
    
    
    HUMAN
    Question: {question} 

    Answer:"""

    prompt_template = PromptTemplate.from_template(prompt)
    
    qa_system_prompt = """You are an assistant for question-answering tasks and write meeting minutes tasks \
    Use the following context to answer question or writing the meeting minutes\
    The answer or meeting minutes should be based on the below context. \
    If the question is not answerable based on the context, you can answer with 'I don't know'.\
    If user ask question, answer the question in concise way.\
    if use ask to write meeting minutes, write the meeting minutes based on the context. \ 

    CONTEXT: {context}\
    """
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )


    chain = prompt_template | llm
    # chain = qa_prompt | llm
    
    for chunks in chain.stream({"chat_history": chat_history,"context": context,  "question": input_text}):
        yield chunks
        








# def keypoints_generator(input_text):
#     chain = key_points_template | llm | StrOutputParser()
#     for chunks in chain.stream( input_text):  
#         yield chunks

def extract_keypoints(transcripts): 
    # Extracting key points
    keypoints = re.findall(r'<kps>(.*?)<\/kps>', transcripts, re.DOTALL)
    # Cleaning up whitespace
    keypoints = [kp.strip() for kp in keypoints]
    return keypoints
    
def verify_question_generator(transcripts):
    st.write(":blue[Now, let me generate questions for these information.]")
    verify_question_prompt = ChatPromptTemplate.from_template("Generate 1 Wh-question for this information: {transcripts}")
    verify_question_chain = verify_question_prompt | llm | StrOutputParser()
    for chunks in verify_question_chain.stream(transcripts):
        yield chunks
# def chain_of_verification(transcripts):

#     for kp in keypoints_generator(transcripts):
#         verify_question = verify_question_generator(kp)
#         for vq in verify_question:
#             yield vq
