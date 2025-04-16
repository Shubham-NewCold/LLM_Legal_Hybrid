# langchain_utils/vectorstore.py
import os
import sys
import traceback
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Optional
from langchain_core.documents import Document

# Add project root to sys.path to allow importing config from the main directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from config import EMBEDDING_MODEL_NAME, EMBEDDING_TRUST_REMOTE_CODE
except ImportError:
    print("ERROR in vectorstore.py: Could not import EMBEDDING_MODEL_NAME from config.py")
    print("Ensure config.py is in the project root directory.")
    # Fallback or exit
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Provide a default fallback
    EMBEDDING_TRUST_REMOTE_CODE = False # Default if import fails
    print(f"WARN: Using default embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"WARN: Using default trust_remote_code: {EMBEDDING_TRUST_REMOTE_CODE}")
    # Or uncomment the next line to stop execution if config is essential
    # sys.exit(1)


# Initialize embedding model using config parameters
# Use a global variable to avoid re-initializing multiple times if possible
embeddings_instance = None

def get_embeddings():
    """Initializes or returns the global embeddings instance."""
    global embeddings_instance
    if embeddings_instance is None:
        print(f"Initializing HuggingFaceEmbeddings with model: {EMBEDDING_MODEL_NAME}...")
        try:
            # trust_remote_code=True might be needed for some models like BGE
            # Set it based on the model you are using. It's generally safer
            # to only set it if required by the specific model documentation.
            trust_remote = True if "bge" in EMBEDDING_MODEL_NAME.lower() else False
            print(f"Using trust_remote_code={EMBEDDING_TRUST_REMOTE_CODE} from config.")
            embeddings_instance = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"trust_remote_code": EMBEDDING_TRUST_REMOTE_CODE} # Set dynamically if needed
            )
            print("Embeddings initialized successfully.")
        except Exception as e:
            print(f"ERROR initializing embeddings with model {EMBEDDING_MODEL_NAME}: {e}")
            traceback.print_exc()
            # Depending on your application's needs, you might want to exit or handle this error
            # sys.exit(1)
            return None # Return None if initialization fails
    return embeddings_instance

# Make embeddings accessible directly if needed by other modules (like qa_chain)
embeddings = get_embeddings()

# --- Specific Functions for Loading and Creating ---

def load_faiss_vectorstore(persist_directory: str) -> Optional[FAISS]:
    """Loads an existing FAISS vectorstore from the specified directory."""
    current_embeddings = get_embeddings()
    if not current_embeddings:
        print("ERROR: Embeddings not initialized. Cannot load vectorstore.")
        return None

    # Check if the core index file exists
    index_file_path = os.path.join(persist_directory, "index.faiss")
    if os.path.exists(index_file_path):
        try:
            print(f"Loading existing FAISS vectorstore from {persist_directory}...")
            vectorstore = FAISS.load_local(
                persist_directory,
                current_embeddings, # Use the initialized embeddings
                allow_dangerous_deserialization=True # Be cautious with this in production
            )
            print("FAISS vectorstore loaded successfully.")
            return vectorstore
        except Exception as e:
            print(f"ERROR loading FAISS index from {persist_directory}: {e}")
            traceback.print_exc()
            return None
    else:
        print(f"WARN: FAISS index file ('index.faiss') not found in {persist_directory}. Cannot load.")
        return None

def create_and_save_faiss_vectorstore(documents: List[Document], persist_directory: str) -> Optional[FAISS]:
    """Creates a new FAISS vectorstore from documents and saves it."""
    current_embeddings = get_embeddings()
    if not current_embeddings:
        print("ERROR: Embeddings not initialized. Cannot create vectorstore.")
        return None
    if not documents:
        print("ERROR: No documents provided to create a new vectorstore.")
        return None

    try:
        print(f"Creating new FAISS vectorstore from {len(documents)} documents...")
        # Ensure the directory exists before saving
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            print(f"Created directory: {persist_directory}")

        vectorstore = FAISS.from_documents(documents, current_embeddings) # Use the initialized embeddings
        print(f"Saving new FAISS vectorstore to {persist_directory}...")
        vectorstore.save_local(persist_directory)
        print(f"New FAISS vectorstore created and saved successfully.")
        return vectorstore
    except Exception as e:
        print(f"ERROR creating or saving new FAISS vectorstore: {e}")
        traceback.print_exc()
        return None

# --- Deprecated Function (Calls Load Only) ---
def initialize_faiss_vectorstore(documents: List[Document], persist_directory: str) -> Optional[FAISS]:
     """
     DEPRECATED - Use load_faiss_vectorstore for loading
     or create_and_save_faiss_vectorstore for building.
     This version attempts to load only. Pass an empty list for documents if only loading.
     """
     print("WARN: Called deprecated initialize_faiss_vectorstore. Attempting to load only.")
     # If documents are passed, it suggests an intent to build/add, which this function no longer does.
     if documents:
         print("WARN: Documents were passed to initialize_faiss_vectorstore, but this function now only loads.")
         print("       Use create_and_save_faiss_vectorstore to build/rebuild the index.")
     return load_faiss_vectorstore(persist_directory)

# Example of how to potentially use this file directly (optional)
if __name__ == '__main__':
    print("\nTesting vectorstore functions...")
    # Assumes config.py defines PERSIST_DIRECTORY in the parent directory
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    persist_dir_test = os.path.join(parent_dir, "faiss_db_test") # Use a test directory

    print(f"Using test directory: {persist_dir_test}")

    # Clean up test directory if it exists
    if os.path.exists(persist_dir_test):
        import shutil
        print("Removing existing test directory...")
        shutil.rmtree(persist_dir_test)

    # Test creation
    print("\n--- Testing Creation ---")
    test_docs = [
        Document(page_content="This is test document one.", metadata={"source": "test1"}),
        Document(page_content="This is test document two.", metadata={"source": "test2"})
    ]
    created_vs = create_and_save_faiss_vectorstore(test_docs, persist_dir_test)
    if created_vs:
        print("Creation test successful.")
    else:
        print("Creation test failed.")

    # Test loading
    print("\n--- Testing Loading ---")
    loaded_vs = load_faiss_vectorstore(persist_dir_test)
    if loaded_vs:
        print("Loading test successful.")
        # Test search
        results = loaded_vs.similarity_search("document two")
        print("Similarity search results for 'document two':")
        for doc in results:
            print(f"- {doc.page_content} (Metadata: {doc.metadata})")
    else:
        print("Loading test failed.")

    # Clean up test directory
    if os.path.exists(persist_dir_test):
        print("\nRemoving test directory...")
        shutil.rmtree(persist_dir_test)