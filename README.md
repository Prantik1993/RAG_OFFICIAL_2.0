# GDPR Legal RAG System v4.2

Production-grade, Advanced RAG pipeline for GDPR (EU 2016/679).
Hybrid retrieval (FAISS + BM25) · CrossEncoder reranking · Prompt versioning · RAGAS evaluation

---

## Architecture — full flow

```
                           QUERY
                             │
                    ┌────────▼─────────┐
                    │   FastAPI :8000   │
                    └────────┬─────────┘
                             │
              ┌──────────────▼──────────────┐
              │      SafetyGuardrails        │
              │  injection · length · garbage│
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │         QueryCache           │
              │     LRU · SHA256 key         │
              └──────────────┬──────────────┘
                        cache MISS
                             │
              ┌──────────────▼──────────────┐
              │       QueryAnalyzer          │  regex only — ZERO LLM cost
              │  EXACT · RANGE · SEMANTIC    │
              └────────┬──────────┬──────────┘
                       │          │
          ┌────────────▼──┐   ┌───▼──────────┐
          │  FAISS (dense) │   │  BM25 (sparse)│  Step 1: RETRIEVE
          │  semantic      │   │  keyword      │  K_FETCH=20 each
          └────────────┬──┘   └───┬───────────┘
                       │          │
              ┌────────▼──────────▼──────┐
              │     RRF Fusion            │  Step 2: FUSE
              │  Reciprocal Rank Fusion   │  ~40 unique candidates
              └──────────────┬────────────┘
                             │
              ┌──────────────▼──────────────┐
              │    CrossEncoder Reranker     │  Step 3: RERANK
              │  ms-marco-MiniLM-L-6-v2     │  scores every (q,doc) pair
              │  22MB · CPU · ~50ms         │  returns final top-K=6
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │     PromptRegistry           │  Step 4: LOAD PROMPT
              │  prompts/vN.yaml · versioned │  active = PROMPT_VERSION env
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │       ChatOpenAI             │  Step 5: GENERATE
              │  gpt-4o-mini · temp=0        │  ONE LLM call per query
              │  + RunnableWithMessageHistory│
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │   Output Safety · Tracker   │
              │   QueryCache.set()          │
              └──────────────┬──────────────┘
                             │
                    ChatResponse
               answer · sources · source_files
               prompt_version · rerank_scores


STARTUP SEQUENCE (once at launch)
───────────────────────────────────
VectorStoreManager.load_or_create()
  ├─ FAISS: load from disk  OR  IngestionPipeline.run()
  │     └─ scans ALL *.pdf in data/pdfs/
  │         GDPRParser — regex hierarchy CHAPTER→SECTION→ARTICLE→POINT→SUBPOINT
  │         stamps every chunk: source_file + source_hash
  │         RecursiveTextSplitter — oversized chunks split
  │         HuggingFaceEmbeddings — all-MiniLM-L6-v2 (CPU, free)
  │         FAISS.from_documents() + save
  └─ BM25: always rebuild in-memory (~1s, no persistence)
PromptRegistry — loads all prompts/vN.yaml
RAGEngine — wires everything together
```

---

## File map

```
legal_rag/
├── prompts/                         Versioned prompt YAMLs
│   ├── v1.yaml                      Strict grounded-only
│   └── v2.yaml                      + structured output + source attribution
├── evaluation/
│   ├── gdpr_testset.json            20 hand-crafted Q&A pairs
│   └── latest_results.json          Auto-generated after eval run
├── data/pdfs/                       Drop all your PDFs here
├── storage/faiss_index/             Auto-created on first run
├── logs/                            app.log + llm_calls.jsonl
└── src/
    ├── config.py                    All settings via .env
    ├── logger.py                    Rotating file + console
    ├── exceptions.py                Domain exceptions per layer
    ├── api.py                       FastAPI — all endpoints
    ├── ui.py                        Streamlit chat UI
    ├── ingestion/
    │   ├── parser.py                Regex hierarchy parser
    │   └── pipeline.py             Multi-PDF · source_file metadata
    ├── retrieval/
    │   ├── query_analyzer.py        Regex intent classifier (no LLM)
    │   ├── bm25_index.py            BM25Okapi keyword index
    │   ├── fusion.py                Reciprocal Rank Fusion
    │   ├── reranker.py              CrossEncoder ms-marco
    │   └── retriever.py             Hybrid FAISS+BM25+RRF+Rerank
    ├── rag/
    │   └── engine.py                One LLM call · prompt registry
    ├── prompts/
    │   └── registry.py             YAML prompt loader · versioning
    ├── vector_store/
    │   └── manager.py               FAISS + BM25 build/load
    ├── evaluation/
    │   └── ragas_eval.py            RAGAS metrics · CLI · compare
    ├── guardrails/safety.py         Input + output validation
    ├── caching/query_cache.py       LRU in-memory cache
    ├── middleware/rate_limiter.py   Per-session token bucket
    └── monitoring/tracker.py        Latency · prompt version log
tests/
    ├── test_query_analyzer.py       18 regex tests
    ├── test_bm25.py                 8 BM25 tests
    ├── test_fusion.py               6 RRF tests
    ├── test_reranker.py             6 CrossEncoder tests
    ├── test_prompt_registry.py      7 prompt registry tests
    ├── test_pipeline_multi_pdf.py   5 multi-PDF tests
    ├── test_evaluation.py           8 RAGAS metric tests
    └── test_api.py                  4 API smoke tests
```

---

## Quick start

```bash
# 1. Install
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env — set OPENAI_API_KEY

# 3. Drop PDFs
cp your_gdpr_docs/*.pdf data/pdfs/

# 4. Run
uvicorn src.api:app --reload          # API  :8000
streamlit run src/ui.py               # UI   :8501

# 5. Evaluate
python -m src.evaluation.ragas_eval                    # all 20 questions
python -m src.evaluation.ragas_eval --quick            # first 5 (fast)
python -m src.evaluation.ragas_eval --compare 1 2      # A/B prompt test

# Docker
docker-compose up --build
```

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Main chat — accepts `prompt_version` param |
| GET | `/prompts` | List all prompt versions |
| GET | `/eval/latest` | Last evaluation results |
| POST | `/eval/run` | Trigger background evaluation |
| GET | `/health` | Server health |
| GET | `/metrics` | LLM call stats by prompt version |
| POST | `/cache/clear` | Clear query cache |

---

## Prompt versioning

Edit `prompts/v2.yaml` or create `prompts/v3.yaml`. Change active via env:

```bash
PROMPT_VERSION=2 uvicorn src.api:app   # use v2
PROMPT_VERSION=latest uvicorn src.api:app  # always highest
```

A/B test two versions:
```bash
python -m src.evaluation.ragas_eval --compare 1 2
```

---

## Cost breakdown

| Component | Cost | Notes |
|---|---|---|
| Embeddings (HuggingFace) | $0 | Local CPU |
| BM25 search | $0 | Pure Python |
| CrossEncoder reranker | $0 | 22MB local model |
| Query analysis | $0 | Regex only |
| **gpt-4o-mini** | ~$0.0002/query | Only cost |
| Cache hits | $0 | Same query free |

500 users × 5 queries/day = **~$0.50/day**

---

## RAG maturity level

| Feature | Status |
|---|---|
| Naive RAG (chunk → embed → search → LLM) | ✅ |
| Structured chunking with metadata | ✅ |
| Hybrid search (dense + sparse) | ✅ |
| RRF fusion | ✅ |
| CrossEncoder reranking | ✅ |
| Prompt versioning + A/B testing | ✅ |
| RAGAS evaluation suite | ✅ |
| Multi-PDF with source attribution | ✅ |
| Query rewriting / HyDE | — next step |
| Self-RAG (grounding verification loop) | — next step |

