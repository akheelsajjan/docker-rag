from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag import vectorstore

files = ["data/apple_fruit.txt", "data/apple_inc.txt", "data/llm_production_guide.txt"]

docs = []
for f in files:
    docs.extend(TextLoader(f).load())

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)

vectorstore.add_documents(chunks)
print(f"Uploaded {len(chunks)} chunks to Pinecone")