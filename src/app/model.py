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
from llmlingua import PromptCompressor

from prompt_template import key_points_template, action_items_template
from prompt_template import mm_template


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
                    temperature=0.1, 
                    top_p=1,
                    mirostat = 2,
                    mirostat_tau =1,
                    )
    return llm
@st.cache_resource
def load_ollama_llamma3_json():
    llm = ChatOllama(model="llama3", 
                    temperature=0.1, 
                    top_p=1,
                    mirostat = 2,
                    mirostat_tau =1,
                    format="json"
                    )
    return llm
llm_json = load_ollama_llamma3_json()
llm = load_ollama_llamma3()
chat_history = []

@st.cache_resource
def load_llllingua():
    llm_lingua = PromptCompressor(
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        use_llmlingua2=True, # Whether to use llmlingua-2
    )
    return llm_lingua

llm_lingua = load_llllingua()


def generate_resvised_meeting_minutes(raw_document, first_meeting_minutes, question_and_answer):
    # Define prompt
    prompt_template = """SYSTEM
    Generate a revised meeting minute based on
    the first meeting minute, and the question/answer below. 
    KEEP the correct informations in the first meeting minute.
    ONLY change if there are any conflicts or errors.
    You should indicate the changes made in the revised meeting minute if needed.:
    
    First meeting minute: {first_meeting_minutes}
    
    Verification Question and Answer: {question_and_answer}
    
    Answer:
    
    Changes made in the revised meeting minute:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    # Define LLM chain
    llm_chain = prompt | llm
    for chunks in llm_chain.stream({"raw_document": raw_document, "first_meeting_minutes": first_meeting_minutes, "question_and_answer": question_and_answer}):
        yield chunks

def answer_question(question, context):
    # Define prompt
    prompt_template = """SYSTEM
    Answer the following question based on the context below, keep the answer concise, less than 3 sentences.
    IF you don't know the answer, just say that you don't know or not mentioned in the context.:
    Context: {context}
    Question: {question}
    Answer:"""
    prompt = PromptTemplate.from_template(prompt_template)
    # Define LLM chain
    llm_chain = prompt | llm
    # Define StuffDocumentsChain
    for chunks in llm_chain.stream({"question": question, "context": context}):
        yield chunks

def verify_question_generator(input_text):
    # Define prompt
    prompt_template = """Write ONLY questions based on the following text:
    {input_text}
    Return in json format:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    # Define LLM chain
    llm_chain = LLMChain(llm=llm_json, prompt=prompt)
    # Define StuffDocumentsChain
    for chunks in llm_chain.stream({"input_text": input_text}):
        yield chunks['text']
        


def extract_fact(input_text):
    prompt_template = """SYSTEM
    Extract the mentioned pieces of information that need to 
    be rechecked from the following text:
    {input_text}
    Return in json format:
    """
    
    messages = [
        HumanMessage(
            content=f"SYSTEM Extract the mentioned pieces of information that need to  \
            be rechecked from the following text: \
            {input_text} \
            Return in json format:"
        )
    ]
    prompt = PromptTemplate.from_template(prompt_template)
    # chain = LLMChain(llm=llm_json, prompt=prompt)
    chain = prompt | llm_json
    for chunks in chain.stream({"input_text": input_text}):
        yield chunks
    # llm_json.invoke(messages)


def generate_first_meeting_minute(transcripts, format = mm_template):
    prompt_template = """SYSTEM
    Write a meeting minute follow-up this format : {format}
    and based on the following conversation:
    Transcripts:
    {transcripts}
    """
    prompt = PromptTemplate.from_template(prompt_template)
    llm_chain =  prompt | llm
    for chunks in llm_chain.stream({"transcripts": transcripts, "format": format}):
        yield chunks
        
def compress_transcript(input_text, rate = 0.9):
    compressed_prompt = llm_lingua.compress_prompt(input_text, instruction="", question="", rate = rate)
    return compressed_prompt['compressed_prompt'], compressed_prompt['rate']

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
    
# def verify_question_generator(transcripts):
#     st.write(":blue[Now, let me generate questions for these information.]")
#     verify_question_prompt = ChatPromptTemplate.from_template("Generate 1 Wh-question for this information: {transcripts}")
#     verify_question_chain = verify_question_prompt | llm | StrOutputParser()
#     for chunks in verify_question_chain.stream(transcripts):
#         yield chunks
# def chain_of_verification(transcripts):

#     for kp in keypoints_generator(transcripts):
#         verify_question = verify_question_generator(kp)
#         for vq in verify_question:
#             yield vq
