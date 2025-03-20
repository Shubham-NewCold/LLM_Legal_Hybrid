import os
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from config import (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, 
                    AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_API_KEY, 
                    TEMPERATURE, MAX_TOKENS, PDF_DIR, MAX_TOKENS_THRESHOLD, PROJECT_NAME)
from langchain_utils.vectorstore import initialize_faiss_vectorstore, embeddings
from document_processing.pdf_extractor import extract_documents_from_pdf
from document_processing.parser import LegalDocumentParser

# Initialize LangSmith tracer and callback manager
tracer = LangChainTracer(project_name=PROJECT_NAME)
callback_manager = CallbackManager([tracer])

# Global variables to be initialized
qa_chain = None

def setup_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 12}
    )
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
        openai_api_key=AZURE_OPENAI_API_KEY,
        temperature=TEMPERATURE,
        model_name=AZURE_OPENAI_DEPLOYMENT_NAME,
        max_tokens=MAX_TOKENS,
        callback_manager=callback_manager,
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
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
                if len(doc_obj.page_content.split()) > MAX_TOKENS_THRESHOLD:
                    hierarchical_docs = parser.parse(doc_obj.page_content, source_name=file, page_number=doc_obj.metadata.get("page_number"))
                    all_documents.extend(hierarchical_docs)
                else:
                    all_documents.append(doc_obj)
    return all_documents

def initialize_app():
    """
    Load documents from PDFs, initialize the vectorstore, and set up the QA chain.
    """
    global qa_chain
    documents = load_all_documents(PDF_DIR)
    print(f"Created {len(documents)} documents from PDFs in '{PDF_DIR}'")
    vectorstore = initialize_faiss_vectorstore(documents)
    qa_chain = setup_qa_chain(vectorstore)
    print("QA chain initialized")