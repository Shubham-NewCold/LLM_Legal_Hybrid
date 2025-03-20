import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import PERSIST_DIRECTORY, EMBEDDING_MODEL_NAME

# Initialize embedding model using config parameters
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def initialize_faiss_vectorstore(documents, persist_directory=PERSIST_DIRECTORY):
    if os.path.exists(persist_directory):
        print("Loading existing FAISS vectorstore...")
        vectorstore = FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Creating new FAISS vectorstore with documents...")
        vectorstore = FAISS.from_documents(documents, embedding=embeddings)
        vectorstore.save_local(persist_directory)
    return vectorstore