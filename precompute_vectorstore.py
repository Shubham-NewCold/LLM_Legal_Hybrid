# precompute_vectorstore.py
import os
from config import PDF_DIR, PERSIST_DIRECTORY
from langchain_utils.qa_chain import load_all_documents
from langchain_utils.vectorstore import initialize_faiss_vectorstore

def precompute():
    if os.path.exists(PERSIST_DIRECTORY):
        print(f"Precomputed vectorstore already exists in {PERSIST_DIRECTORY}.")
    else:
        print("Precomputing vectorstore from PDFs...")
        # Process all PDFs to create Document objects
        documents = load_all_documents(PDF_DIR)
        # Build the FAISS vectorstore and save it locally
        initialize_faiss_vectorstore(documents, persist_directory=PERSIST_DIRECTORY)
        print("Vectorstore precomputed and saved.")

if __name__ == "__main__":
    precompute()
