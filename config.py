import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Flask configuration
DEBUG = False
PORT = int(os.environ.get("PORT", 5000))

# Directory settings
PDF_DIR = "pdfs"
PERSIST_DIRECTORY = "faiss_db"

# Model and API settings
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://nec-us2-ai.openai.azure.com/")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini-legal")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# LangChain tracing and callback settings
PROJECT_NAME = "pr-new-molecule-89"
TEMPERATURE = 0.15
MAX_TOKENS = 1024

# Token thresholds for hierarchical parsing
MAX_TOKENS_THRESHOLD = 250
CHUNK_MAX_TOKENS = 200
OVERLAP_RATIO = 0.3