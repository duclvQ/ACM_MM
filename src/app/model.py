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
from langchain_core.prompts import ChatPromptTemplate
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
                    temperature=0.3, 
                    top_p=1,
                    mirostat = 2,
                    mirostat_tau =1,
                    )
    return llm

llm = load_ollama_llamma3()

def meeting_minutes_generator(input_text):
    # convert to string
    input_text = str(input_text)
    docs = text_splitter.create_documents([input_text])
    split_docs = text_splitter.split_documents(docs)
    # # chain = keypoints_template | llm | StrOutputParser()
    # prompt_template = """Write a concise summary topic of the following meeting conversation:
    # {text}
    # CONCISE SUMMARY in bullet points format, keep person name:"""
    # prompt = PromptTemplate.from_template(prompt_template)

    # refine_template = (
    #     "Your job is to produce a final summary, each topic information should be a bullet point.\n"
    #     "We have provided an existing summary up to a certain point: {existing_answer}\n"
    #     "We have the opportunity to refine the existing summary and collapse the overlapping points\n"
    #     "(only if needed) with some more context below.\n"
    #     "------------\n"
    #     "{text}\n"
    #     "------------\n"
    #     "Given the new context, refine the original summary in a way that makes it more accurate or useful.\n"
    #     "If the context isn't useful, return the original summary."
    # )
    # refine_prompt = PromptTemplate.from_template(refine_template)
    # chain = load_summarize_chain(
    #     llm=llm,
    #     chain_type="refine",
    #     question_prompt=prompt,
    #     refine_prompt=refine_prompt,
    #     return_intermediate_steps=True,
    #     input_key="input_documents",
    #     output_key="output_text",
        
    # )
    # Run chain
    # Define prompt
    prompt_template = """Write a list of key points of the following meeting conversation in form of a meeting minutes:
    "{text}"
     key points in bullet points format:"""
    prompt = PromptTemplate.from_template(prompt_template)
    # Define LLM chain
    
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    
    for chunks in stuff_chain.stream(split_docs):
        yield chunks['output_text']

def keypoints_generator(input_text):
    chain = key_points_template | llm | StrOutputParser()
    for chunks in chain.stream( input_text):  
        yield chunks

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
def chain_of_verification(transcripts):

    for kp in keypoints_generator(transcripts):
        verify_question = verify_question_generator(kp)
        for vq in verify_question:
            yield vq
def meeting_minutes_prompt(transcripts):
    return f"Return the keypoints of meeting below: {transcripts}"