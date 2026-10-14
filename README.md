# PDF Semantic Search

Retrieval pipeline over a folder of PDF documents, built during an internship.
PDFs are split into overlapping chunks, embedded with Azure OpenAI (`text-embedding-3-large`)
and stored in [Qdrant](https://qdrant.tech/). Queries return the most relevant passages,
optionally expanded with neighbouring chunks from the same page.

## Project structure

```
.
├── main.py              # CLI entry point
└── rag/
    ├── config.py        # Settings loaded from environment variables
    ├── documents.py     # PDF text extraction and chunking
    ├── embeddings.py    # Azure OpenAI embeddings with on-disk cache
    └── vector_store.py  # Qdrant collection management and search
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then fill in your credentials
```

## Usage

Index every PDF under `./pdf` (sub-folders included). This recreates the collection:

```bash
python main.py ingest --pdf-dir ./pdf
```

Search with a free-text question:

```bash
python main.py search "your question here" --top-k 5
python main.py search "your question here" --folder subfolder-name --no-expand
```

Search using the first question in an Excel column:

```bash
python main.py excel questions.xlsx --sheet Sheet1 --column E
```

Create payload indexes on an existing collection (already done by `ingest`):

```bash
python main.py create-indexes
```

## Configuration

| Variable | Description | Default |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI v1 endpoint | required |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI key | required |
| `QDRANT_URL` | Qdrant cluster URL | required |
| `QDRANT_API_KEY` | Qdrant API key | empty |
| `COLLECTION_NAME` | Qdrant collection | `documents` |
| `PDF_DIR` | Source folder for PDFs | `./pdf` |
| `EMBEDDING_CACHE_PATH` | Local embedding cache | `.cache/embeddings.pkl` |

Embeddings are cached locally so a failed upload can be retried without paying for them again.
Delete the cache file after changing the source documents.

## Context

Developed during an internship. Project-specific data, documents and
credentials have been removed.
