# build_indices.py

import os
import sys
import pickle
import shutil
from typing import List, Dict, Tuple # Added Dict, Tuple
import traceback
import torch # Added torch

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Ensure project root is searched first

# --- Local Imports ---
# --- Updated Config Imports ---
from config import (
    PDF_DIR, PERSIST_DIRECTORY,
    SPLADE_MODEL_NAME, SPLADE_VECTORS_FILENAME, SPLADE_DOCS_FILENAME # Use SPLADE config
)
from langchain_utils.vectorstore import create_and_save_faiss_vectorstore
from document_processing.pdf_extractor import extract_documents_from_pdf
from document_processing.parser import pyparse_hierarchical_chunk_text
from langchain_core.documents import Document

# --- NEW: Import Transformers ---
try:
    from transformers import AutoModelForMaskedLM, AutoTokenizer
except ImportError:
    print("ERROR: transformers library not found. Please install it: pip install transformers torch")
    sys.exit(1)

# --- REMOVED: BM25 Import ---
# try:
#     from rank_bm25 import BM25Okapi
# except ImportError:
#     print("ERROR: rank_bm25 library not found. Please install it: pip install rank_bm25")
#     sys.exit(1)

# Define paths relative to this script's location or use absolute paths
PDF_DIR_ABS = os.path.abspath(os.path.join(os.path.dirname(__file__), PDF_DIR))
PERSIST_DIRECTORY_ABS = os.path.abspath(os.path.join(os.path.dirname(__file__), PERSIST_DIRECTORY))

# --- NEW: SPLADE Index Paths ---
SPLADE_VECTORS_PATH = os.path.join(PERSIST_DIRECTORY_ABS, SPLADE_VECTORS_FILENAME)
SPLADE_DOCS_PATH = os.path.join(PERSIST_DIRECTORY_ABS, SPLADE_DOCS_FILENAME)

# --- REMOVED: BM25 Paths ---
# BM25_INDEX_PATH = os.path.join(PERSIST_DIRECTORY_ABS, BM25_INDEX_FILENAME)
# BM25_DOCS_PATH = os.path.join(PERSIST_DIRECTORY_ABS, BM25_DOCS_FILENAME)

CUSTOMER_LIST_FILE = "detected_customers.txt"
CUSTOMER_LIST_PATH = os.path.join(project_root, CUSTOMER_LIST_FILE)


# --- Document Loading and Parsing Logic (Unchanged from your provided code) ---
def load_all_documents_for_indexing(pdf_directory):
    """Loads PDFs, extracts pages, parses hierarchically for indexing."""
    all_final_documents = []
    overall_chunk_index = 0 # Keep track if needed, though not strictly used later

    if not os.path.isdir(pdf_directory):
        print(f"ERROR: PDF directory not found: {pdf_directory}")
        return []

    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files in {pdf_directory}")

    for file in pdf_files:
        file_path = os.path.join(pdf_directory, file)
        print(f"\n--- Processing {file} ---") # More concise logging
        try:
            # This function now extracts customer name using LLM and adds to metadata
            page_documents = extract_documents_from_pdf(file_path)
            if not page_documents:
                 print(f"WARN: No documents extracted from {file}. Skipping.")
                 continue
        except Exception as e:
            print(f"ERROR: Failed to extract pages from {file}: {e}")
            traceback.print_exc()
            continue

        current_hierarchy_stack = []
        # Process each page document extracted from the PDF
        for doc_index, doc_obj in enumerate(page_documents):
            page_content = doc_obj.page_content
            page_metadata = doc_obj.metadata
            source_file = page_metadata.get('source', file) # Should always be set by extractor
            page_number = page_metadata.get('page_number', doc_index + 1) # Use index as fallback
            # Customer name is already in metadata from pdf_extractor
            customer_name = page_metadata.get('customer', 'Unknown Customer')
            region_name = page_metadata.get('region', 'Unknown Region')
            word_count = len(page_content.split())

            # Prepare metadata for the parser, ensuring customer name is included
            parser_metadata = {
                'source': source_file,
                'customer': customer_name, # Pass the extracted/normalized name
                'region': region_name,
                'clause': 'N/A', # Default clause, parser will update if header found
                'hierarchy': [] # Default hierarchy, parser will update
            }
            # Merge any other metadata from the original page document
            parser_metadata.update({k: v for k, v in page_metadata.items() if k not in parser_metadata})

            try:
                from config import MAX_TOKENS_THRESHOLD as MTT
            except ImportError:
                MTT = 350 # Use default from config if import fails
                print(f"WARN: MAX_TOKENS_THRESHOLD not found in config, using default {MTT}")

            # Decide whether to parse hierarchically based on word count
            if word_count > MTT:
                # print(f"DEBUG: Page {page_number} word count ({word_count}) > {MTT}. Parsing hierarchically.") # Optional Debug
                try:
                    # Parse the page content into smaller chunks
                    parsed_page_docs, current_hierarchy_stack = pyparse_hierarchical_chunk_text(
                        full_text=page_content,
                        source_name=source_file,
                        page_number=page_number,
                        extra_metadata=parser_metadata, # Pass combined metadata
                        initial_stack=current_hierarchy_stack # Maintain hierarchy across pages
                    )
                    # Add the resulting chunks (which inherit/update metadata)
                    all_final_documents.extend(parsed_page_docs)
                except Exception as e:
                    print(f"ERROR: Failed to parse page {page_number} of {file}: {e}")
                    traceback.print_exc()
                    print(f"  WARNING: Adding page {page_number} as whole chunk due to parsing error.")
                    # Add the original page document as a single chunk, ensuring metadata is consistent
                    doc_obj.metadata = parser_metadata # Use the prepared metadata
                    doc_obj.metadata['page_number'] = page_number
                    # Try to add current hierarchy state even on error
                    doc_obj.metadata['hierarchy'] = [item[0] for item in current_hierarchy_stack] if current_hierarchy_stack else []
                    doc_obj.metadata['clause'] = current_hierarchy_stack[-1][0] if current_hierarchy_stack else 'N/A'
                    all_final_documents.append(doc_obj)
            else:
                # Page is short enough, add as a single chunk
                # print(f"DEBUG: Page {page_number} word count ({word_count}) <= {MTT}. Adding as whole chunk.") # Optional Debug
                doc_obj.metadata = parser_metadata # Use the prepared metadata
                doc_obj.metadata['page_number'] = page_number
                # Add current hierarchy state
                doc_obj.metadata['hierarchy'] = [item[0] for item in current_hierarchy_stack] if current_hierarchy_stack else []
                doc_obj.metadata['clause'] = current_hierarchy_stack[-1][0] if current_hierarchy_stack else 'N/A'
                all_final_documents.append(doc_obj)

    print(f"\nTotal documents processed into chunks: {len(all_final_documents)}")
    return all_final_documents


# --- NEW: SPLADE Indexing Function ---
def build_and_save_splade(
    documents: List[Document],
    model,
    tokenizer,
    vectors_save_path: str,
    docs_save_path: str,
    batch_size: int = 4 # Adjust based on GPU memory
) -> Tuple[List[Dict[int, float]], List[Document]]:
    """Builds and saves SPLADE sparse vectors and the corresponding documents."""
    if not documents:
        print("ERROR: No documents provided to build SPLADE index.")
        return None, None

    print(f"\nPreparing and encoding data for SPLADE from {len(documents)} chunks...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval() # Set model to evaluation mode

    splade_vectors = []
    splade_docs = documents # Keep the full Document objects

    doc_contents = [doc.page_content for doc in splade_docs]

    for i in range(0, len(doc_contents), batch_size):
        batch_texts = doc_contents[i:i+batch_size]
        tokens = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)

        with torch.no_grad():
            # Get the MLM logits, aggregate over sequence length using max pooling
            # Compute the vector ('v') representation for the batch
            doc_reps = model(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])[0] # [batch, seq_len, vocab_size]
            # Use ReLU activation and max pooling over sequence length
            doc_reps_activated = torch.relu(doc_reps)
            pooled_reps, _ = torch.max(doc_reps_activated * tokens['attention_mask'].unsqueeze(-1), dim=1) # [batch, vocab_size]
            # Apply log saturation
            pooled_reps_log_sat = torch.log1p(pooled_reps)

        # Convert to sparse dictionary format
        batch_vectors = pooled_reps_log_sat.cpu().numpy()
        for vec in batch_vectors:
            indices = vec.nonzero()[0]
            weights = vec[indices]
            sparse_dict = dict(zip(indices.tolist(), weights.tolist()))
            splade_vectors.append(sparse_dict)

        print(f"  Encoded batch {i//batch_size + 1}/{(len(doc_contents) + batch_size - 1)//batch_size}")

    print(f"SPLADE encoding complete. Generated {len(splade_vectors)} sparse vectors.")

    try:
        # Save the list of sparse vector dictionaries
        with open(vectors_save_path, "wb") as f:
            pickle.dump(splade_vectors, f)
        print(f"SPLADE sparse vectors saved to {vectors_save_path}")

        # Save the list of Document objects used
        with open(docs_save_path, "wb") as f:
            pickle.dump(splade_docs, f)
        print(f"SPLADE document list saved to {docs_save_path}")

        return splade_vectors, splade_docs
    except Exception as e:
        print(f"ERROR saving SPLADE vectors or documents: {e}")
        traceback.print_exc()
        return None, None

# --- REMOVED: BM25 Indexing Function ---
# def build_and_save_bm25(...)

# --- Main Indexing Logic ---
if __name__ == "__main__":
    print("Starting indexing process...")
    rebuild = len(sys.argv) > 1 and sys.argv[1] == '--rebuild'

    # --- Handle rebuild ---
    if rebuild and os.path.exists(PERSIST_DIRECTORY_ABS):
        print(f"Rebuild requested. Removing existing index directory: {PERSIST_DIRECTORY_ABS}")
        try:
            shutil.rmtree(PERSIST_DIRECTORY_ABS)
            print("Existing index directory removed.")
        except Exception as e:
            print(f"ERROR removing existing index directory: {e}")
            sys.exit(1) # Exit if removal fails

    if not os.path.exists(PERSIST_DIRECTORY_ABS):
        try:
            os.makedirs(PERSIST_DIRECTORY_ABS)
            print(f"Created index directory: {PERSIST_DIRECTORY_ABS}")
        except Exception as e:
            print(f"ERROR creating index directory: {e}")
            sys.exit(1) # Exit if creation fails

    # --- Check if indices exist (unless rebuilding) ---
    faiss_exists = os.path.exists(os.path.join(PERSIST_DIRECTORY_ABS, "index.faiss"))
    # --- NEW: Check SPLADE files ---
    splade_vectors_exist = os.path.exists(SPLADE_VECTORS_PATH)
    splade_docs_exist = os.path.exists(SPLADE_DOCS_PATH)

    # --- REMOVED: BM25 file checks ---
    # bm25_model_exists = os.path.exists(BM25_INDEX_PATH)
    # bm25_docs_exist = os.path.exists(BM25_DOCS_PATH)

    # --- Updated Check ---
    if not rebuild and faiss_exists and splade_vectors_exist and splade_docs_exist:
        print("Indices (FAISS & SPLADE) already exist. Use '--rebuild' argument to force recreation.")
        sys.exit(0)

    # --- 1. Load and process documents ---
    print(f"\nLoading documents from: {PDF_DIR_ABS}")
    final_chunks = load_all_documents_for_indexing(PDF_DIR_ABS)

    if not final_chunks:
        print("ERROR: No documents were processed. Exiting.")
        sys.exit(1)

    # --- 2. Build and save FAISS index ---
    if not faiss_exists or rebuild:
        print("\nBuilding FAISS index...")
        vectorstore = create_and_save_faiss_vectorstore(final_chunks, persist_directory=PERSIST_DIRECTORY_ABS)
        if not vectorstore:
            print("ERROR: FAISS index creation failed.")
            sys.exit(1)
    else:
        print("\nFAISS index already exists. Skipping build.")

    # --- 3. Build and save SPLADE index ---
    if not (splade_vectors_exist and splade_docs_exist) or rebuild:
        print("\nLoading SPLADE model for indexing...")
        try:
            splade_tokenizer = AutoTokenizer.from_pretrained(SPLADE_MODEL_NAME)
            splade_model = AutoModelForMaskedLM.from_pretrained(SPLADE_MODEL_NAME)
            print(f"SPLADE model '{SPLADE_MODEL_NAME}' loaded.")

            # Pass final_chunks (which are Document objects)
            build_and_save_splade(
                final_chunks,
                model=splade_model,
                tokenizer=splade_tokenizer,
                vectors_save_path=SPLADE_VECTORS_PATH,
                docs_save_path=SPLADE_DOCS_PATH
            )
            # Add check if build_and_save_splade failed
            if not os.path.exists(SPLADE_VECTORS_PATH) or not os.path.exists(SPLADE_DOCS_PATH):
                 print("ERROR: SPLADE index creation/saving failed.")
                 sys.exit(1)
        except Exception as e:
            print(f"ERROR during SPLADE model loading or indexing: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\nSPLADE index already exists. Skipping build.")

    # --- REMOVED: BM25 build section ---

    # --- 4. Update detected_customers.txt (Unchanged) ---
    print(f"\nUpdating customer list file: {CUSTOMER_LIST_PATH}")
    detected_customers = set()
    for doc in final_chunks:
        # Extract customer name from metadata, default to "Unknown Customer"
        customer_name = doc.metadata.get('customer', 'Unknown Customer')
        # Add to set if it's a valid name (not empty and not the default unknown)
        if customer_name and customer_name != 'Unknown Customer':
            detected_customers.add(customer_name)

    if detected_customers:
        # Sort the unique names alphabetically for consistency
        sorted_customers = sorted(list(detected_customers))
        try:
            # Open the file in write mode ('w'), overwriting existing content
            with open(CUSTOMER_LIST_PATH, 'w') as f:
                for name in sorted_customers:
                    f.write(name + '\n') # Write each name on a new line
            print(f"Successfully updated {CUSTOMER_LIST_FILE} with {len(sorted_customers)} unique customer names:")
            for name in sorted_customers: print(f"  - {name}") # Log the names written
        except Exception as e:
            print(f"ERROR writing to {CUSTOMER_LIST_FILE}: {e}")
            traceback.print_exc()
            # Decide if this error should stop the whole process
            # sys.exit(1)
    else:
        print(f"WARN: No valid customer names found in processed documents. {CUSTOMER_LIST_FILE} not updated.")
        # Optionally clear the file if no customers are found
        try:
            with open(CUSTOMER_LIST_PATH, 'w') as f:
                pass # Creates an empty file or clears existing one
            print(f"Cleared content of {CUSTOMER_LIST_FILE} as no customers were detected.")
        except Exception as e:
            print(f"ERROR clearing {CUSTOMER_LIST_FILE}: {e}")
    # --- End of new block ---

    print("\nIndexing process complete.")