import time
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from rag.config import (
    COLLECTION_NAME,
    EMBEDDING_DIM,
    QDRANT_API_KEY,
    QDRANT_TIMEOUT,
    UPSERT_BATCH_SIZE,
    UPSERT_MAX_RETRIES,
    require_env,
)
from rag.embeddings import embed_query

PAYLOAD_INDEXES = {
    "document": PayloadSchemaType.KEYWORD,
    "page": PayloadSchemaType.INTEGER,
    "folder": PayloadSchemaType.KEYWORD,
}


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=require_env("QDRANT_URL"), api_key=QDRANT_API_KEY or None, timeout=QDRANT_TIMEOUT)


def create_payload_indexes(client: QdrantClient) -> None:
    for field_name, schema in PAYLOAD_INDEXES.items():
        client.create_payload_index(COLLECTION_NAME, field_name=field_name, field_schema=schema)


def recreate_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    create_payload_indexes(client)


def upsert_chunks(client: QdrantClient, chunks: list[dict], embeddings: list[list[float]]) -> None:
    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{c['document']}-{c['page']}-{c['chunk_index']}")),
            vector=vector,
            payload=c,
        )
        for c, vector in zip(chunks, embeddings)
    ]

    total = len(points)
    for i in range(0, total, UPSERT_BATCH_SIZE):
        batch = points[i:i + UPSERT_BATCH_SIZE]
        for attempt in range(1, UPSERT_MAX_RETRIES + 1):
            try:
                client.upsert(collection_name=COLLECTION_NAME, points=batch)
                break
            except Exception as e:
                print(f"  Batch {i}-{i + len(batch)} failed (attempt {attempt}/{UPSERT_MAX_RETRIES}): {e}")
                if attempt == UPSERT_MAX_RETRIES:
                    raise
                time.sleep(2 * attempt)
        print(f"  uploaded {min(i + UPSERT_BATCH_SIZE, total)}/{total}")


def search(client: QdrantClient, question: str, top_k: int = 5, folder: str | None = None) -> list[ScoredPoint]:
    query_filter = None
    if folder:
        query_filter = Filter(must=[FieldCondition(key="folder", match=MatchValue(value=folder))])

    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=embed_query(question),
        limit=top_k,
        query_filter=query_filter,
    ).points


def expand_context(client: QdrantClient, result: ScoredPoint, window: int = 1) -> str:
    """Return the matched chunk joined with its neighbours on the same page."""
    p = result.payload
    wanted = range(max(0, p["chunk_index"] - window), p["chunk_index"] + window + 1)

    neighbors, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(must=[
            FieldCondition(key="document", match=MatchValue(value=p["document"])),
            FieldCondition(key="page", match=MatchValue(value=p["page"])),
        ]),
        limit=50,
        with_payload=True,
    )

    relevant = sorted(
        (n for n in neighbors if n.payload["chunk_index"] in wanted),
        key=lambda n: n.payload["chunk_index"],
    )
    return " ".join(n.payload["text"] for n in relevant)
