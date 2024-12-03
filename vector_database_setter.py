import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from openai import OpenAI

# How to perform business

def load_texts(directory):
    texts = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
    return texts

def query_documents(query, k=3):
    results = vectordb.similarity_search(query, k=k)
    return results


persist_directory = "chromadb"

# Load documents
documents = load_texts('./archive/business')

# Initialize embedding function
embeddings = OpenAIEmbeddings()

# Create Chroma vector store
vectordb = Chroma.from_texts(
    texts=documents,
    embedding=embeddings,
    persist_directory=persist_directory
)

# Persist the vector store
vectordb.persist()

print("Documents successfully added to ChromaDB.")
