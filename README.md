# GDPR Legal RAG System v4.1  —  Hybrid + Rerank

Production-grade RAG pipeline for GDPR (EU 2016/679) Q&A.

## Architecture

```
                        QUERY
                          │
                    [FastAPI :8000]
                          │
               ┌──────────▼──────────┐
               │  SafetyGuardrails   │  ← injection, length, garbage check
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │    QueryCache       │  ← LRU in-memory, SHA256 key
               └──────────┬──────────┘
                    cache MISS
                          │
               ┌──────────▼──────────┐
               │   QueryAnalyzer     │  ← regex only, zero LLM cost
               │  (intent + refs)    │    EXACT / RANGE / SEMANTIC
               └──────┬────────┬─────┘
                       │        │
              ┌────────▼──┐  ┌──▼──────────┐
              │   FAISS   │  │    BM25      │   STEP 1: RETRIEVE
              │  (dense)  │  │  (sparse)    │   fetch K_FETCH=20 each
              │ semantic  │  │  keyword     │
              └────────┬──┘  └──┬───────────┘
                       │        │
               ┌───────▼────────▼──────┐
               │  RRF Fusion           │   STEP 2: FUSE
               │  (Reciprocal Rank     │   rank-merge without score normalisation
               │   Fusion, k=60)       │   deduplicates, ~40 unique candidates
               └──────────┬────────────┘
                          │
               ┌──────────▼──────────┐
               │  CrossEncoder       │   STEP 3: RERANK
               │  Reranker           │   scores every (query, doc) pair jointly
               │  ms-marco-MiniLM    │   returns final top-K=6
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │  ChatOpenAI         │   STEP 4: GENERATE (1 LLM call)
               │  gpt-4o-mini        │   strict grounded-only prompt
               │  + chat history     │   RunnableWithMessageHistory
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │  output safety      │  ← prompt leak detection
               │  + LLMTracker       │  ← latency JSONL log
               │  + QueryCache.set() │
               └──────────┬──────────┘
                          │
                    ChatResponse
                  answer + sources
                  + rerank_scores


STARTUP (once on server launch)
────────────────────────────────
VectorStoreManager.load_or_create()
  ├─ FAISS: load from disk  OR  build from Documents
  └─ BM25:  always rebuild in-memory (~1s, no persistence needed)

IngestionPipeline.run()  (only if no FAISS index on disk)
  GDPRParser.parse()       regex → LegalChunks (CHAPTER→SECTION→ARTICLE→POINT→SUBPOINT)
  RecursiveTextSplitter    oversized chunks split, metadata preserved
  HuggingFaceEmbeddings    all-MiniLM-L6-v2, runs on CPU
  FAISS.from_documents()   build + save to storage/faiss_index/
```

## File map

```
src/
├── config.py                   All settings via .env
├── logger.py                   Rotating file + console
├── exceptions.py               Domain exceptions per layer
├── api.py                      FastAPI — /chat /health /metrics /cache/clear
├── ui.py                       Streamlit chat UI
├── ingestion/
│   ├── parser.py               Deterministic GDPR hierarchy parser
│   └── pipeline.py             Parse → split → Documents
├── retrieval/
│   ├── query_analyzer.py       Regex intent classifier (no LLM)
│   ├── bm25_index.py           BM25Okapi keyword index        ← NEW
│   ├── fusion.py               Reciprocal Rank Fusion         ← NEW
│   ├── reranker.py             CrossEncoder reranker          ← NEW
│   └── retriever.py            Hybrid FAISS+BM25+RRF+Rerank   ← UPDATED
├── rag/
│   └── engine.py               One LLM call per query
├── vector_store/
│   └── manager.py              FAISS + BM25 build/load        ← UPDATED
├── guardrails/safety.py        Input + output validation
├── caching/query_cache.py      LRU in-memory cache
├── middleware/rate_limiter.py  Per-session token bucket
└── monitoring/tracker.py       Latency + call log
tests/
├── test_query_analyzer.py      Regex parser (no deps)
├── test_bm25.py                BM25 index (no deps)           ← NEW
├── test_fusion.py              RRF logic (no deps)            ← NEW
├── test_reranker.py            CrossEncoder (downloads model) ← NEW
└── test_api.py                 FastAPI smoke tests
```

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env              # set OPENAI_API_KEY
cp your_gdpr.pdf data/pdfs/CELEX_32016R0679_EN_TXT.pdf

uvicorn src.api:app --reload      # API  :8000
streamlit run src/ui.py           # UI   :8501

# or
docker-compose up --build
```

## Tests

```bash
pytest tests/ -v                       # all
pytest tests/test_bm25.py -v           # BM25 only (fast, no downloads)
pytest tests/test_fusion.py -v         # RRF only  (fast, no downloads)
pytest tests/test_reranker.py -v       # needs model download ~22 MB
```

## What changed in v4.1

| v4.0                          | v4.1                                      |
|-------------------------------|-------------------------------------------|
| FAISS semantic only           | FAISS + BM25 hybrid                       |
| No fusion                     | Reciprocal Rank Fusion (RRF)              |
| No reranking                  | CrossEncoder ms-marco-MiniLM-L-6-v2       |
| Single retrieval path         | EXACT / RANGE / SEMANTIC with per-intent fusion |
| rerank_score not exposed      | Returned in API response metadata         |

## Env vars

| Variable            | Default                                  | Description                     |
|---------------------|------------------------------------------|---------------------------------|
| OPENAI_API_KEY      | (required)                               |                                 |
| LLM_MODEL           | gpt-4o-mini                              | LLM for generation              |
| EMBEDDING_MODEL     | sentence-transformers/all-MiniLM-L6-v2   | Bi-encoder for FAISS            |
| RERANKER_MODEL      | cross-encoder/ms-marco-MiniLM-L-6-v2     | CrossEncoder for reranking      |
| RETRIEVAL_K         | 6                                        | Final docs sent to LLM          |
| RETRIEVAL_K_FETCH   | 20                                       | Candidates per retriever        |
| CHUNK_SIZE          | 1500                                     |                                 |
| CHUNK_OVERLAP       | 200                                      |                                 |
| RATE_LIMIT_RPM      | 15                                       |                                 |
| RATE_LIMIT_RPH      | 200                                      |                                 |
| CACHE_MAX_SIZE      | 1000                                     |                                 |
