import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

INDEX_NAME = "langsmith-llm-prod-guide"



def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


def ingest(files: list[str]):
    """One-time ingestion of source docs into Pinecone."""
    docs = []
    for f in files:
        docs.extend(TextLoader(f).load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vectorstore.add_documents(chunks)
    print(f"Uploaded {len(chunks)} chunks to Pinecone")


if __name__ == "__main__":
    ingest([
        "data/apple_fruit.txt",
        "data/apple_inc.txt",
        "data/llm_production_guide.txt",
    ])