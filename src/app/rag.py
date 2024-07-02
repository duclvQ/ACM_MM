from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.tools import tool
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.documents import Document
from typing import List

def create_retriever(text_file_path, chunk_size = 300, \
                    chunk_overlap = 0, separator = "\n", \
                    reset=True, model_name="all-MiniLM-L6-v2"):
    # Load the document, split it into chunks, embed each chunk and load it into the vector store.
    raw_documents = TextLoader(text_file_path).load()
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=separator)
    documents = text_splitter.split_documents(raw_documents)
    embedding_function = SentenceTransformerEmbeddings(model_name=model_name)
    # reset the vector store
    if reset:
        db = Chroma()
        all_ids = db.get()
        for id in all_ids['ids']:
            db.delete(id)
    # # load the document into Chroma
    db = Chroma.from_documents(documents, embedding_function)
    retriever = db.as_retriever()
    return retriever


class quoted_answer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based on the given sources,",
    )
    citations: str = Field(
        ..., description="Citations from the given sources and should keep the relevant content in that source, DO NOT paraphrase the source."
    )
    
def format_docs_with_id(docs: List[Document]) -> str:
    formatted = [
        f"Source ID: {i},\n{doc.page_content}"
        for i, doc in enumerate(docs)
    ]
    return "\n\n" + "\n\n".join(formatted)