import os

from dotenv import load_dotenv

load_dotenv()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "3072"))

QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "120"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")

PDF_DIR = os.getenv("PDF_DIR", "./pdf")
EMBEDDING_CACHE_PATH = os.getenv("EMBEDDING_CACHE_PATH", ".cache/embeddings.pkl")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBED_BATCH_SIZE = 100
UPSERT_BATCH_SIZE = 64
UPSERT_MAX_RETRIES = 3
