import pickle
from functools import lru_cache
from pathlib import Path

from openai import OpenAI

from rag.config import EMBED_BATCH_SIZE, EMBEDDING_CACHE_PATH, EMBEDDING_MODEL, require_env


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    return OpenAI(
        base_url=require_env("AZURE_OPENAI_ENDPOINT"),
        api_key=require_env("AZURE_OPENAI_API_KEY"),
    )


def embed_texts(texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> list[list[float]]:
    client = get_openai_client()
    embeddings = []
    for i in range(0, len(texts), batch_size):
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts[i:i + batch_size])
        embeddings.extend(item.embedding for item in response.data)
        print(f"  embedded {min(i + batch_size, len(texts))}/{len(texts)}")
    return embeddings


def embed_query(text: str) -> list[float]:
    response = get_openai_client().embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def embed_chunks_cached(chunks: list[dict], cache_path: str = EMBEDDING_CACHE_PATH) -> list[list[float]]:
    """Reuse embeddings from disk so a failed upload doesn't require re-embedding everything."""
    path = Path(cache_path)
    if path.exists():
        with path.open("rb") as f:
            cached = pickle.load(f)
        if len(cached) == len(chunks):
            print(f"Loaded embeddings from cache '{path}'")
            return cached
        print("Cache size doesn't match current chunks, recomputing...")

    embeddings = embed_texts([c["text"] for c in chunks])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings cached to '{path}'")
    return embeddings
