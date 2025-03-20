import os
import pymupdf4llm
from langchain_core.documents import Document

def extract_documents_from_pdf(pdf_path):
    """
    Extract documents from a PDF using PyMuPDF4LLM and enrich metadata with customer info.
    """
    file_name = os.path.basename(pdf_path)
    if "lactalis" in file_name.lower():
        customer = "Lactalis Australia"
        region = "Australia"
    elif "patties" in file_name.lower():
        customer = "Patties Foods Pty Ltd"
        region = "Australia"
    elif "simplot" in file_name.lower():
        customer = "Simplot Australia Pty Limited"
        region = "Australia"
    elif "fonterra" in file_name.lower():
        customer = "Fonterra"
        region = "New Zealand"
    else:
        customer = "Unknown Customer"
        region = "Unknown Region"

    # Convert the PDF to Markdown with page chunks
    md_chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    documents = []
    for idx, chunk in enumerate(md_chunks):
        text = chunk.get("text", "").strip()
        metadata = chunk.get("metadata", {})
        metadata["source"] = file_name
        metadata["customer"] = customer
        metadata["region"] = region
        if "page_number" not in metadata:
            metadata["page_number"] = idx + 1
        documents.append(Document(page_content=text, metadata=metadata))
    return documents