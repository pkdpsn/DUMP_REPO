import asyncio
import time
from typing import List

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# Config
# -----------------------------
MAX_BATCH_SIZE = 32          # process at most 32 requests at once
MAX_BATCH_DELAY = 0.5        # wait up to 0.5 sec for batching
DEVICE = "cuda"              # GPU

# -----------------------------
# Model (SentenceTransformer reranker)
# -----------------------------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(DEVICE)

# -----------------------------
# Request + Response Schemas
# -----------------------------
class RerankRequest(BaseModel):
    query: str
    candidates: List[str]

# -----------------------------
# Async Queue For Requests
# -----------------------------
queue: asyncio.Queue = asyncio.Queue()

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI()

@app.post("/rerank")
async def rerank(req: RerankRequest):
    """
    Push request into queue and wait for batch worker to return result.
    """
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    # Put the incoming request + future into queue
    await queue.put((req, future))

    # Wait for the batched result
    result = await future
    return result


# -----------------------------
# Batching Worker
# -----------------------------
async def batch_worker():
    """
    Continuously pull requests from asyncio.Queue and batch them.
    """
    while True:
        reqs = []
        futures = []

        # Wait until at least one item exists
        req, fut = await queue.get()
        reqs.append(req)
        futures.append(fut)
        start_time = time.time()

        # Try to collect more up to batch size or timeout
        while len(reqs) < MAX_BATCH_SIZE and (time.time() - start_time) < MAX_BATCH_DELAY:
            try:
                req2, fut2 = queue.get_nowait()
                reqs.append(req2)
                futures.append(fut2)
            except asyncio.QueueEmpty:
                # No more immediate requests → wait a tiny bit
                await asyncio.sleep(0.005)

        # -----------------------------
        # Run batched inference on GPU
        # -----------------------------
        results = await run_batched_rerank(reqs)

        # Send each response to its request via its future
        for fut, result in zip(futures, results):
            fut.set_result(result)


async def run_batched_rerank(requests: List[RerankRequest]):
    """
    Vectorized reranking for multiple requests at once.
    """
    # Flatten all queries and candidates across the batch
    all_queries = [r.query for r in requests]
    all_candidates = [r.candidates for r in requests]

    # Encode queries and candidate lists
    # ----- Encode queries -----
    q_embeddings = model.encode(
        all_queries,
        convert_to_tensor=True,
        device=DEVICE,
        show_progress_bar=False
    )

    # Encode candidates (batch dynamic)
    cand_embeddings = []
    for cand_list in all_candidates:
        emb = model.encode(
            cand_list,
            convert_to_tensor=True,
            device=DEVICE,
            show_progress_bar=False
        )
        cand_embeddings.append(emb)

    # ----- Compute reranking -----
    responses = []
    for i, req in enumerate(requests):
        q = q_embeddings[i]
        cands = cand_embeddings[i]

        # cosine similarity
        scores = util.cos_sim(q, cands)[0]

        # sort descending
        sorted_idx = torch.argsort(scores, descending=True)
        ranked = [req.candidates[j] for j in sorted_idx]
        ranked_scores = [float(scores[j]) for j in sorted_idx]

        responses.append({
            "query": req.query,
            "reranked": ranked,
            "scores": ranked_scores
        })

    return responses


# -----------------------------
# Start the batching worker
# -----------------------------
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(batch_worker())
