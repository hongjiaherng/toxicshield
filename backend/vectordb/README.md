# VectorDB Module

Hybrid vector database for toxicity classification using dense + sparse embeddings, backed by Qdrant.

## Architecture

```
┌───────────────────────────────────────────────────┐
│                    VectorDB                       │
│                                                   │
│  cloud_inference=False          cloud_inference=True
│  ┌───────────────────┐         ┌────────────────┐ │
│  │ Local Embeddings  │         │ Qdrant Cloud   │ │
│  │ (fastembed)       │         │ Inference      │ │
│  │                   │         │                │ │
│  │ • Dense: MiniLM   │         │ • models.Document│
│  │ • Sparse: BM25    │         │   (text sent as- │
│  └───────┬───────────┘         │   is to Qdrant)│ │
│          │                     └───────┬────────┘ │
│          ▼                             ▼          │
│        PointStruct              PointStruct       │
│          └──────────┬──────────────┘              │
│                     ▼                             │
│            Qdrant Cloud Cluster                   │
│        Collection: toxicity_reference             │
│                                                   │
│  Search: dense / sparse / hybrid (RRF)            │
└───────────────────────────────────────────────────┘
```

## Configuration

| Constant | Default | Description |
|---|---|---|
| `COLLECTION_NAME` | `toxicity_reference` | Qdrant collection name |
| `DENSE_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Dense embedding model |
| `DENSE_EMBEDDING_SIZE` | `384` | Dense vector dimension |
| `SPARSE_MODEL` | `Qdrant/bm25` | Sparse embedding model |

Environment variables (`.env`):

| Variable | Description |
|---|---|
| `QDRANT_API_KEY` | Qdrant Cloud API key |
| `QDRANT_ENDPOINT` | Qdrant Cloud cluster URL |

## Quick Start

```python
from vectordb import VectorDB

db = VectorDB(cloud_inference=False)  # or True for cloud-side inference
db.setup()

# Insert samples
db.insert([
    {"text": "You are terrible", "is_toxic": True, "category": "Insults and Flaming"},
    {"text": "Great work!", "is_toxic": False, "category": "Non-Toxic"},
])

# Search
results = db.hybrid_search("you are a failure", k=3)
# → [{"text": ..., "category": ..., "score": ...}, ...]
```

## API Reference

### `VectorDB(cloud_inference=False)`

Initializes the database client and optionally loads local embedding models.

- **`cloud_inference=False`** — Embeddings are computed locally via `fastembed`. Requires sufficient RAM to load models.
- **`cloud_inference=True`** — Raw text is sent to Qdrant Cloud, which handles embedding internally via `models.Document`. No local models loaded.

### `setup()`

Creates the `toxicity_reference` collection if it doesn't exist, configured with:

- **Dense vectors** (`dense`) — cosine similarity, 384 dimensions
- **Sparse vectors** (`sparse`) — BM25, in-memory index
- **Payload indexes** — `is_toxic` (bool), `category` (keyword)

Idempotent — safe to call on every startup.

### `insert(samples: list[dict])`

Upserts documents into the collection.

**Sample format:**

```python
{"text": str, "is_toxic": bool, "category": str}
```

Points are assigned sequential integer IDs starting from `0`. If you need stable IDs across runs, assign them yourself before calling insert (or extend `_build_points`).

### `dense_search(query_text, k=3) -> list[dict]`

Searches using dense embeddings only. Filters to `is_toxic=True` results.

Returns: `[{"text": str, "category": str, "score": float}, ...]`

### `sparse_search(query_text, k=3) -> list[dict]`

Searches using sparse (BM25) embeddings only. Filters to `is_toxic=True` results.

### `hybrid_search(query_text, k=3) -> list[dict]`

Performs hybrid search using Reciprocal Rank Fusion (RRF) over both dense and sparse prefetches (20 candidates each). This is the recommended search method — it combines semantic understanding (dense) with keyword matching (sparse).

Filters to `is_toxic=True` results only.

## Search Strategy Comparison

| Method | Strengths | Weaknesses |
|---|---|---|
| `dense_search` | Semantic matching, handles paraphrases | Misses exact keyword matches |
| `sparse_search` | Exact keyword/term matching | No semantic understanding |
| `hybrid_search` | Best of both via RRF fusion | Slightly higher latency |

## Payload Schema

Every point in the collection stores:

```json
{
  "text": "the original text",
  "is_toxic": true,
  "category": "Insults and Flaming"
}
```

Categories for toxic samples: `Insults and Flaming`, `Hate and Harassment`, `Threats`, `Extremism`, `Other Offensive Texts`. Non-toxic samples use `Non-Toxic`.

## Dependencies

- `qdrant-client` — Qdrant client library
- `fastembed` — Local embedding generation (only needed when `cloud_inference=False`)
- `python-dotenv` — Environment variable loading

## Logging

Logs are written to `backend/log/vectordb.log` (rotating, 5MB max) and to stdout.
