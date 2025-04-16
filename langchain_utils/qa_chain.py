# langchain_utils/qa_chain.py
import os
import sys
import pickle
from typing import Any, Dict, List, Optional, Sequence, Union
import traceback
import copy
import torch # Added torch

# --- LangChain Imports ---
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.mapreduce import MapReduceDocumentsChain
# --- LLM Imports ---
# from langchain_openai import AzureChatOpenAI # Keep if you might switch back
from langchain_google_genai import ChatGoogleGenerativeAI # USE THIS
# --- Core Imports ---
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain.globals import set_debug
from langchain.schema.vectorstore import VectorStoreRetriever

# --- Reranker Import ---
try:
    from sentence_transformers.cross_encoder import CrossEncoder
except ImportError:
    print("ERROR: sentence-transformers library not found. Reranking will be disabled.")
    print("Please install it: pip install -U sentence-transformers")
    CrossEncoder = None # Define as None if import fails

# --- NEW: Transformers Import ---
try:
    from transformers import AutoModelForMaskedLM, AutoTokenizer
except ImportError:
    print("ERROR: transformers library not found. Please install it: pip install transformers torch")
    AutoModelForMaskedLM, AutoTokenizer = None, None # Define as None if import fails

# --- Local Imports ---
# Ensure the project root is in the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Updated Config Imports ---
from config import (
    TEMPERATURE, MAX_TOKENS, PERSIST_DIRECTORY,
    PROJECT_NAME, RETRIEVAL_TYPE, TOP_K,
    # SPLADE Config
    SPLADE_MODEL_NAME, SPLADE_VECTORS_FILENAME, SPLADE_DOCS_FILENAME, SPLADE_TOP_N,
    # Reranker Config
    RERANKER_ENABLED, RERANKER_MODEL_NAME,
    # Azure Config (Remove Google API Key)
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT_NAME
)
try:
    from config import SIMILARITY_SCORE_THRESHOLD
except ImportError:
    SIMILARITY_SCORE_THRESHOLD = 0.7

# --- Vectorstore Import ---
from langchain_utils.vectorstore import load_faiss_vectorstore, embeddings

# --- System Prompt Import ---
try:
    from system_prompt import system_prompt
    print("--- Successfully imported system_prompt from system_prompt.py ---")
except ImportError:
    print("--- WARNING: Could not import system_prompt. Using a basic default for Reduce step. ---")
    # Define a basic fallback if system_prompt.py is missing
    system_prompt = """
You are a helpful AI assistant. Synthesize the provided summaries to answer the question based *only* on the summaries. Attribute information using the metadata (Source, Customer, Clause). If the question asks for specific information (e.g., 'chilled temperature'), only include data explicitly matching that criteria from the summaries. If no matching information is found for a requested entity, state that clearly.
"""


# --- Global Variables ---
map_reduce_chain: Optional[MapReduceDocumentsChain] = None
vectorstore = None
retriever: Optional[VectorStoreRetriever] = None
llm_instance: Optional[AzureChatOpenAI] = None # LLM for MapReduce
detected_customer_names: List[str] = []
CUSTOMER_LIST_FILE = "detected_customers.txt"

# --- NEW: SPLADE Globals ---
splade_model = None
splade_tokenizer = None
splade_vectors: List[Dict[int, float]] = []
splade_docs: List[Document] = []

# --- REMOVED: BM25 Globals ---
# bm25_index = None
# bm25_docs: List[Document] = []

# --- Reranker Global ---
reranker_model = None
# RERANKER_MODEL_NAME is imported from config

# --- Check Azure Credentials (Done in config.py, but good practice) ---
if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME]):
    print("CRITICAL ERROR in qa_chain.py: Azure OpenAI credentials missing in config. LLM initialization will fail.")
    # sys.exit(1) # Optional exit


# --- MapReduce Chain Setup ---
def setup_map_reduce_chain() -> MapReduceDocumentsChain:
    global llm_instance
    if llm_instance is None:
        # Check credentials again before instantiation
        if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME]):
             raise ValueError("Azure OpenAI credentials not found in config. Cannot initialize LLM.")
        try:
            # --- Initialize Azure OpenAI ---
            print(f"Initializing AzureChatOpenAI (Deployment: {AZURE_OPENAI_DEPLOYMENT_NAME})...") # Update log
            llm_instance = AzureChatOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME, # Use azure_deployment
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS, # Use max_tokens
                # model_name=AZURE_OPENAI_MODEL_NAME, # Optional if deployment implies model
            )
            print("AzureChatOpenAI initialized successfully.") # Update log
        except Exception as e:
             print(f"ERROR initializing AzureChatOpenAI: {e}") # Update log
             traceback.print_exc()
             sys.exit(1) # Exit if LLM fails to initialize

    llm = llm_instance # Use the initialized Azure LLM

    # --- Stricter Map Prompt ---
    map_template = """
You will be provided with a document excerpt preceded by its source metadata (Source, Page, Customer, Clause).
Your task is to analyze ONLY the text of the document excerpt BELOW the '---' line.
Based *only* on this excerpt text, identify and extract the exact sentences or key points that are relevant to answering the user's question.

User Question: {question}

Document Excerpt with Metadata:
{page_content}

**Critical Instructions:**
1.  Focus *only* on the text provided in the excerpt *below* the '---' line.
2.  Extract verbatim sentences or concise key points from the excerpt text that *directly provide information relevant to the User Question's topic(s) and entities*.
3.  **Pay special attention to any specific details, entities, figures, conditions, requirements, or keywords mentioned *in the User Question*. Extract the exact text from the excerpt that contains or directly addresses these specifics.**
4.  **Handling Comparative Questions:** If the User Question asks for a comparison between entities (e.g., 'compare A and B about topic X'), your task in *this step* is to extract any information about **topic X** that relates to **either** entity A **or** entity B, *if that information is present in this specific excerpt*. Do not discard relevant information about entity A just because entity B is not mentioned here, or vice-versa. The final comparison synthesis will happen in a later step based on all extractions.
5.  Use the context of the surrounding text in the excerpt to determine relevance, even if the specific keyword from the question isn't in the exact sentence being extracted.
6.  Extract information regardless of formatting (e.g., inside lists, tables, or ```code blocks```).
7.  **Your entire output MUST start with the *exact* metadata line provided above (everything before the '---'), followed by ' --- ', and then either your extracted text OR the specific phrase \"No relevant information found in this excerpt.\"**
8.  If the excerpt text contains NO information relevant *at all* to the **topic(s) or target entities** mentioned in the User Question (considering instruction #4 for comparisons), your output MUST be the metadata line followed by ' --- No relevant information found in this excerpt.'.
9.  Do NOT add explanations, introductions, summaries, or any text other than the required metadata prefix and the extracted relevant text (or the \"No relevant information\" message).
10. Do NOT attempt to answer the overall User Question.

**Output:**
"""

    map_prompt = PromptTemplate(
        input_variables=["page_content", "question"],
        template=map_template
    )
    # Use the same LLM for map and reduce unless performance/cost dictates otherwise
    map_chain = LLMChain(llm=llm, prompt=map_prompt, verbose=True)

    # --- Reduce Prompt (Uses system_prompt) ---
    reduce_template = f"""
{system_prompt}

Original User Question: {{question}}

Extracted Information Summaries (Metadata --- Content):
{{doc_summaries}}

Based *only* on the summaries above and following all instructions in the initial system prompt (especially regarding strict grounding, requested entities, and specific query terms like 'chilled'), provide the final answer to the Original User Question. If multiple summaries provide relevant details for the same point, synthesize them concisely. If summaries indicate no specific information was found for a requested entity or criteria (e.g., 'chilled temperature for Patties'), explicitly state that in the final answer.

Final Answer:"""
    reduce_prompt = PromptTemplate(
        input_variables=["doc_summaries", "question"],
        template=reduce_template
        )
    reduce_llm_chain = LLMChain(llm=llm, prompt=reduce_prompt, verbose=True)

    # --- Combine Setup ---
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_llm_chain,
        document_variable_name="doc_summaries",
        document_separator="\n\n---\n\n", # Separator used between map outputs
        verbose=True
    )

    # --- MapReduce Chain ---
    chain = MapReduceDocumentsChain(
        llm_chain=map_chain, # The chain for the map step
        reduce_documents_chain=combine_documents_chain, # The chain for the reduce step
        document_variable_name="page_content", # Input variable name in map_prompt
        input_key="input_documents", # Input key for the overall chain
        output_key="output_text", # Output key for the overall chain
        verbose=True
    )
    return chain


# --- Application Initialization ---
def initialize_app():
    """Initializes vectorstore, retriever, SPLADE, reranker, chain, and detected customer names."""
    # Add SPLADE globals
    global vectorstore, retriever, map_reduce_chain, detected_customer_names
    global splade_model, splade_tokenizer, splade_vectors, splade_docs # NEW
    global reranker_model, llm_instance
    set_debug(True) # Or set based on config

    print("Initializing application components...")

    # --- Load FAISS Vectorstore (Unchanged) ---
    if not os.path.exists(PERSIST_DIRECTORY) or not os.listdir(PERSIST_DIRECTORY):
         print(f"CRITICAL ERROR: FAISS index directory '{PERSIST_DIRECTORY}' not found or empty.")
         print("Please run 'python build_indices.py' first.")
         sys.exit(1)
    vectorstore = load_faiss_vectorstore(persist_directory=PERSIST_DIRECTORY)
    if not vectorstore:
        print("CRITICAL ERROR: Failed to load FAISS vectorstore. Exiting.")
        sys.exit(1)
    print("FAISS vectorstore loaded.")

    # --- NEW: Load SPLADE Model and Data ---
    splade_vectors_path = os.path.join(PERSIST_DIRECTORY, SPLADE_VECTORS_FILENAME)
    splade_docs_path = os.path.join(PERSIST_DIRECTORY, SPLADE_DOCS_FILENAME)
    if AutoModelForMaskedLM is None: # Check if transformers loaded
        print("CRITICAL ERROR: Transformers library not loaded. Cannot initialize SPLADE.")
        sys.exit(1)
    try:
        print(f"Loading SPLADE model: {SPLADE_MODEL_NAME}...")
        splade_tokenizer = AutoTokenizer.from_pretrained(SPLADE_MODEL_NAME)
        splade_model = AutoModelForMaskedLM.from_pretrained(SPLADE_MODEL_NAME)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        splade_model.to(device)
        splade_model.eval() # Set to evaluation mode
        print(f"SPLADE model '{SPLADE_MODEL_NAME}' loaded successfully to {device}.")

        if os.path.exists(splade_vectors_path) and os.path.exists(splade_docs_path):
            print(f"Loading SPLADE vectors from {splade_vectors_path}...")
            with open(splade_vectors_path, "rb") as f: splade_vectors = pickle.load(f)
            with open(splade_docs_path, "rb") as f: splade_docs = pickle.load(f)
            print(f"SPLADE vectors and documents loaded successfully ({len(splade_docs)} documents).")
        else:
            print(f"WARN: SPLADE vectors or document list not found in {PERSIST_DIRECTORY}.")
            print("SPLADE search will not be available. Please run 'python build_indices.py'.")
            splade_vectors, splade_docs = [], [] # Ensure lists are empty
    except Exception as e:
        print(f"ERROR loading SPLADE model or data: {e}")
        traceback.print_exc()
        # Set to None/empty on error to allow readiness check to fail
        splade_model, splade_tokenizer, splade_vectors, splade_docs = None, None, [], []


    # --- REMOVED: Load BM25 Index and Documents ---
    # bm25_index_path = os.path.join(PERSIST_DIRECTORY, BM25_INDEX_FILENAME)
    # bm25_docs_path = os.path.join(PERSIST_DIRECTORY, BM25_DOCS_FILENAME)
    # if os.path.exists(bm25_index_path) and os.path.exists(bm25_docs_path):
    #     print(f"Loading BM25 index from {bm25_index_path}...")
    #     try:
    #         with open(bm25_index_path, "rb") as f: bm25_index = pickle.load(f)
    #         with open(bm25_docs_path, "rb") as f: bm25_docs = pickle.load(f)
    #         print(f"BM25 index and documents loaded successfully ({len(bm25_docs)} documents).")
    #     except Exception as e:
    #         print(f"ERROR loading BM25 index or documents: {e}")
    #         bm25_index, bm25_docs = None, []
    # else:
    #     print(f"WARN: BM25 index or document list not found in {PERSIST_DIRECTORY}.")
    #     print("BM25 search will not be available. Please run 'python build_indices.py'.")
    #     bm25_index, bm25_docs = None, []

    # --- Load Detected Customer Names (Unchanged) ---
    try:
        customer_list_path = os.path.join(project_root, CUSTOMER_LIST_FILE) # Use absolute path
        with open(customer_list_path, "r") as f:
            # Use set for uniqueness, then sort
            names = sorted(list(set(line.strip() for line in f if line.strip() and line.strip() != "Unknown Customer")))
            detected_customer_names = names
        print(f"Loaded {len(detected_customer_names)} unique customer names from {customer_list_path}.")
    except FileNotFoundError:
        print(f"WARN: {CUSTOMER_LIST_FILE} not found at {project_root}. Customer name list is empty.")
        detected_customer_names = []
    except Exception as e:
        print(f"Error loading {CUSTOMER_LIST_FILE}: {e}")
        detected_customer_names = []

    # --- Retriever Setup (FAISS - Unchanged) ---
    if vectorstore:
        try:
            search_kwargs = {}
            if RETRIEVAL_TYPE == "similarity_score_threshold":
                search_kwargs['score_threshold'] = SIMILARITY_SCORE_THRESHOLD
                search_kwargs['k'] = TOP_K # Also pass K for threshold search
                print(f"Initializing FAISS retriever: type='similarity_score_threshold', threshold={SIMILARITY_SCORE_THRESHOLD}, k={TOP_K}")
            else: # Default to similarity (top-k)
                search_kwargs['k'] = TOP_K
                print(f"Initializing FAISS retriever: type='similarity', k={TOP_K}")

            retriever = vectorstore.as_retriever(
                search_type=RETRIEVAL_TYPE,
                search_kwargs=search_kwargs
            )
            print("FAISS retriever initialized successfully.")
        except Exception as e:
             print(f"CRITICAL ERROR creating FAISS retriever: {e}")
             traceback.print_exc()
             sys.exit(1)
    else:
        # This case should have been caught by vectorstore loading check
        print("CRITICAL ERROR: Vectorstore not available. Cannot create retriever.")
        sys.exit(1)

    # --- Load Reranker Model (Unchanged) ---
    if RERANKER_ENABLED:
        if CrossEncoder:
            try:
                print(f"Loading Reranker model: {RERANKER_MODEL_NAME}...")
                reranker_model = CrossEncoder(RERANKER_MODEL_NAME, max_length=512, trust_remote_code=True) # Default to CPU/auto
                print(f"Reranker model '{RERANKER_MODEL_NAME}' loaded successfully.")
            except Exception as e:
                print(f"ERROR loading Reranker model '{RERANKER_MODEL_NAME}': {e}")
                print("Reranking will be disabled.")
                traceback.print_exc()
                reranker_model = None # Ensure it's None on error
        else:
            print("WARN: CrossEncoder class not available (sentence-transformers not installed?). Reranking disabled.")
            reranker_model = None # Ensure it's None if import failed
    else:
        print("INFO: Reranking is disabled in config.")
        reranker_model = None # Ensure it's None if disabled

    # --- Chain Setup (Calls setup_map_reduce_chain which now uses Azure) ---
    try:
        # setup_map_reduce_chain will initialize llm_instance if it's None
        map_reduce_chain = setup_map_reduce_chain()
        print("MapReduce chain initialized successfully (using Azure OpenAI).")
    except Exception as e:
        print(f"CRITICAL ERROR setting up MapReduce chain: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("Application initialization complete.")

# --- Function to get detected names (Unchanged) ---
def get_detected_customer_names() -> List[str]:
    """Returns the list of customer names detected during initialization."""
    global detected_customer_names
    # No warning needed here, just return the current state
    return detected_customer_names

# --- Direct Execution Test Block (Optional - Unchanged) ---
# if __name__ == '__main__':
#     print("Running qa_chain.py directly...")
#     # Add any test logic here if needed, e.g., initialize and test chain components
#     try:
#         initialize_app()
#         print("Initialization successful in direct run.")
#         # Example test:
#         if map_reduce_chain and retriever:
#             print("Chain and retriever seem loaded.")
#         else:
#             print("Chain or retriever failed to load.")
#     except SystemExit:
#         print("Initialization failed, exiting.")
#     except Exception as e:
#         print(f"An error occurred during direct run test: {e}")