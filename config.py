# config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Flask configuration
DEBUG = False # Keep False for production/deployment
PORT = int(os.environ.get("PORT", 5000))

# Directory settings
PDF_DIR = "pdfs"
PERSIST_DIRECTORY = "faiss_db" # Base directory for indices

# --- NEW: SPLADE Index Filenames ---
SPLADE_VECTORS_FILENAME = "splade_vectors.pkl" # Filename for pickled sparse vectors
SPLADE_DOCS_FILENAME = "splade_docs.pkl"    # Filename for the list of Document objects used by SPLADE

# --- REMOVED: BM25 Filenames ---
# BM25_INDEX_FILENAME = "bm25_index.pkl"
# BM25_DOCS_FILENAME = "bm25_docs.pkl"

# --- Core Model Settings ---

# Embedding Model (for FAISS - Dense Retrieval)
EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v3"
EMBEDDING_TRUST_REMOTE_CODE = True
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Alternative smaller model

# --- NEW: SPLADE Model (for Sparse Retrieval) ---
# Choose a SPLADE model from Hugging Face. naver/splade-cocondenser-ensembledistil is efficient.
SPLADE_MODEL_NAME = "naver/splade-v3"
# Other options: "naver/splade-cocondenser-selfdistil", "naver/splade-v2-distil", etc.

# LLM Settings (Using AzureOpenAI via API Key)
TEMPERATURE = 0.2 # Low temperature for factual consistency
MAX_TOKENS = 2048 # Max tokens for LLM responses (Map/Reduce steps)

# --- Optional Azure OpenAI Settings (Keep if needed as fallback/alternative) ---
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://nec-us2-ai.openai.azure.com/")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-35-turbo")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# LangChain tracing and callback settings
PROJECT_NAME = "pr-new-molecule-89" # Optional: for LangSmith tracing

# Token thresholds for hierarchical parsing
MAX_TOKENS_THRESHOLD = 350 # Max words before hierarchical parsing is triggered

# --- Retrieval Settings ---
# Configure FAISS retriever type and parameters (Dense Retrieval)
RETRIEVAL_TYPE = "similarity" # Using top-k for the dense part of hybrid search
TOP_K = 100 # How many results to initially get from FAISS

# --- NEW: SPLADE Top-N (Sparse Retrieval) ---
SPLADE_TOP_N = 100 # How many results to initially get from SPLADE

# --- REMOVED: BM25 Top-N ---
# BM25_TOP_N = 50

# Settings for Hybrid Search (RRF)
RRF_K = 60 # Constant for Reciprocal Rank Fusion (default = 60)

# --- Reranking Settings ---
RERANKER_ENABLED = True # Set to False to disable reranking easily
RERANKER_MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"
RERANK_CANDIDATE_POOL_SIZE = 200 # How many candidates from RRF to feed into the reranker
PASS_2_SCORE_THRESHOLD = 0.3 # Optional threshold for comparative query Pass 2 selection

# --- LLM Chain Input Limit Settings ---
# HYBRID_TOP_K = 12 # <<< Keep this commented out or remove if using Dynamic K primarily

# --- NEW: Dynamic K Settings ---
DYNAMIC_K_ENABLED = True # Set to True to enable dynamic K calculation
# Threshold for combined score (0.0 to 1.0). Tune based on score distribution.
DYNAMIC_K_SCORE_THRESHOLD = 0.4
# Minimum number of chunks to send to LLM, even if few meet threshold.
DYNAMIC_K_MIN_CHUNKS = 1   
# Maximum number of chunks to send to LLM, even if many meet threshold.
DYNAMIC_K_MAX_CHUNKS = 12   

# Add Azure Checks
if not AZURE_OPENAI_ENDPOINT:
    print("CRITICAL WARNING: AZURE_OPENAI_ENDPOINT is not set in the environment variables (.env file). Azure LLM will not work.")
if not AZURE_OPENAI_API_KEY:
    print("CRITICAL WARNING: AZURE_OPENAI_API_KEY is not set in the environment variables (.env file). Azure LLM will not work.")
if not AZURE_OPENAI_DEPLOYMENT_NAME:
    print("CRITICAL WARNING: AZURE_OPENAI_DEPLOYMENT_NAME is not set in the environment variables (.env file). Azure LLM will not work.")

print(f"--- Config Loaded ---")
print(f"LLM Provider: Azure OpenAI")
print(f"  Endpoint Set: {'Yes' if AZURE_OPENAI_ENDPOINT else 'NO'}")
print(f"  API Key Set: {'Yes' if AZURE_OPENAI_API_KEY else 'NO'}")
print(f"  Deployment: {AZURE_OPENAI_DEPLOYMENT_NAME if AZURE_OPENAI_DEPLOYMENT_NAME else 'NOT SET'}")
print(f"  API Version: {AZURE_OPENAI_API_VERSION}")
print(f"Dense Embedding Model (FAISS): {EMBEDDING_MODEL_NAME}")
print(f"Sparse Model (SPLADE): {SPLADE_MODEL_NAME}")
print(f"Reranker Enabled: {RERANKER_ENABLED}")
if RERANKER_ENABLED:
    print(f"Reranker Model: {RERANKER_MODEL_NAME}")
    print(f"Rerank Candidate Pool Size: {RERANK_CANDIDATE_POOL_SIZE}")
    print(f"Smart Select Pass 2 Score Threshold: {PASS_2_SCORE_THRESHOLD if PASS_2_SCORE_THRESHOLD is not None else 'Disabled'}")

# Updated print statement for K value
if DYNAMIC_K_ENABLED:
    print(f"Dynamic K Enabled: True (Threshold={DYNAMIC_K_SCORE_THRESHOLD}, Min={DYNAMIC_K_MIN_CHUNKS}, Max={DYNAMIC_K_MAX_CHUNKS})")
else:
    # If dynamic K is disabled, use DYNAMIC_K_MAX_CHUNKS as the fixed K value
    # (which defaults to HYBRID_TOP_K if dynamic settings weren't imported)
    fixed_k_value = DYNAMIC_K_MAX_CHUNKS
    print(f"Dynamic K Enabled: False. Using Fixed K = {fixed_k_value}")

print(f"--------------------")