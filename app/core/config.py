import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Fixed PDF file (hardcoded path inside repo)
PDF_PATH = "data/kau-medicinal.pdf"

# Chroma persistent directory (hardcoded)
CHROMA_DIR = "chroma_store"
CHROMA_COLLECTION = "fixed_pdf_docs"

# Chunking params
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150
