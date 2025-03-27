import os
import sys
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from email_tracer import EmailLangChainTracer
from langchain_core.callbacks.manager import CallbackManager
from langchain.callbacks.tracers.langchain import LangChainTracer

from config import (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, 
                    AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_API_KEY, 
                    TEMPERATURE, MAX_TOKENS, PDF_DIR, MAX_TOKENS_THRESHOLD, PROJECT_NAME, PERSIST_DIRECTORY, OPENROUTER_API_KEY)
from langchain_utils.vectorstore import initialize_faiss_vectorstore, embeddings
from document_processing.pdf_extractor import extract_documents_from_pdf
from document_processing.parser import LegalDocumentParser
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your system prompt from the project root
from system_prompt import system_prompt


# Global variables to be initialized
qa_chain = None

def setup_qa_chain(vectorstore, top_k_vectors=22):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k_vectors}
    )

    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
        openai_api_key=AZURE_OPENAI_API_KEY,
        temperature=TEMPERATURE,
        model_name=AZURE_OPENAI_DEPLOYMENT_NAME,
        max_tokens=MAX_TOKENS,
        #callback_manager=callback_manager,
    )
    
    # Create a custom prompt template that integrates the system prompt.
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=f"""{system_prompt}

Context:
{{context}}

User Query:
{{question}}

Answer:"""
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": custom_prompt}
    )

def load_all_documents(pdf_directory):
    from langchain_core.documents import Document
    all_documents = []
    for file in os.listdir(pdf_directory):
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(pdf_directory, file)
            print(f"Processing {file_path}...")
            page_documents = extract_documents_from_pdf(file_path)
            parser = LegalDocumentParser()
            for doc_obj in page_documents:
                # Pass extra metadata (which includes customer, region, etc.) to the parser.
                extra_metadata = doc_obj.metadata
                if len(doc_obj.page_content.split()) > MAX_TOKENS_THRESHOLD:
                    hierarchical_docs = parser.parse(doc_obj.page_content, source_name=file, page_number=doc_obj.metadata.get("page_number"), extra_metadata=extra_metadata)
                    all_documents.extend(hierarchical_docs)
                else:
                    all_documents.append(doc_obj)
    return all_documents

def initialize_app():
    global vectorstore, qa_chain
    if os.path.exists(PERSIST_DIRECTORY):
        print("Loading precomputed FAISS vectorstore...")
        # Pass an empty list for documents, assuming the vectorstore loader in your utility
        # simply loads from disk when files are present.
        vectorstore = initialize_faiss_vectorstore([], persist_directory=PERSIST_DIRECTORY)
    else:
        print("Precomputed vectorstore not found; building from scratch...")
        documents = load_all_documents(PDF_DIR)
        vectorstore = initialize_faiss_vectorstore(documents, persist_directory=PERSIST_DIRECTORY)
    qa_chain = setup_qa_chain(vectorstore)
    print("QA chain initialized")