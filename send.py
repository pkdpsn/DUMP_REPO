reranker_service/
│
├── app/
│   ├── main.py                  # FastAPI app + lifecycle
│   ├── settings.py              # env / config
│
│   ├── api/
│   │   ├── rerank.py             # inference endpoint
│   │   ├── health.py             # /health, /ready
│
│   ├── core/
│   │   ├── model.py              # model loader
│   │   ├── batcher.py            # async batching engine
│   │   ├── metrics.py            # splunk-compatible metrics
│   │   ├── logging.py            # structured logging
│   │
│   ├── schemas/
│   │   ├── request.py
│   │   ├── response.py
│
│   ├── workers/
│   │   └── rerank_worker.py      # batch consumer
│
├── requirements.txt
└── README.md



Client
  │
  ▼
FastAPI
  │
  ├── Request Validation (Pydantic)
  ├── Async Queue (batcher)
  │        │
  │        ▼
  │   Batch Worker (background task)
  │        │
  │        ▼
  │   Reranker Model
  │
  ├── Metrics (latency, batch size)
  ├── Health / Readiness
  └── Logs → Splunk



Problem Restated (in system terms)

Reranker

hot path

high QPS

low latency

must stay on GPU all day

Embedding model

cold / bursty

latency tolerant

used only certain hours

should not occupy GPU memory when idle

Goal:

Keep reranker always hot, dynamically load/unload embedding model.


UNLOADED → LOADING → ACTIVE → IDLE → EVICTED

models:
  - name: reranker
    type: reranker
    device: cuda:0
    priority: high
    batching:
      enabled: true
      max_batch_size: 16
      max_wait_ms: 10
    eviction:
      enabled: false

  - name: embedding
    type: embedding
    device: cuda:0
    priority: low
    batching:
      enabled: true
      max_batch_size: 64
      max_wait_ms: 50
    eviction:
      enabled: true
      idle_timeout_sec: 1800   # 30 minutes
      offload_to: cpu
Lazy Loading (on first request)
