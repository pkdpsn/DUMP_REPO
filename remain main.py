import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from sentence_transformers import SentenceTransformer, util
import time

app = FastAPI()

class RerankInput(BaseModel):
    query: str
    documents: list[str]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to("cuda")

QUEUE = asyncio.Queue()
MAX_BATCH_SIZE = 32
MAX_WAIT_TIME = 0.5  # seconds

async def batch_worker():
    while True:
        start_time = time.time()
        batch = []
        futures = []

        # Wait for first item
        item = await QUEUE.get()
        batch.append(item["req"])
        futures.append(item["future"])

        # Keep adding items until timeout OR batch is full
        while len(batch) < MAX_BATCH_SIZE:
            wait_time = MAX_WAIT_TIME - (time.time() - start_time)
            if wait_time <= 0:
                break
            try:
                item = await asyncio.wait_for(QUEUE.get(), timeout=wait_time)
                batch.append(item["req"])
                futures.append(item["future"])
            except asyncio.TimeoutError:
                break

        # ---- RUN GPU INFERENCE HERE ----
        queries = [b.query for b in batch]
        docs = [b.documents for b in batch]

        # Flatten all docs for fast embedding
        all_docs = [d for sub in docs for d in sub]

        query_embeddings = model.encode(queries, convert_to_tensor=True)
        doc_embeddings = model.encode(all_docs, convert_to_tensor=True)

        # Compute cosine similarities
        results = []
        idx = 0
        for i, doc_set in enumerate(docs):
            num_docs = len(doc_set)
            doc_embeds_slice = doc_embeddings[idx: idx + num_docs]
            idx += num_docs
            scores = util.cos_sim(query_embeddings[i], doc_embeds_slice)
            results.append(scores.tolist())

        # Return results to individual requests
        for fut, res in zip(futures, results):
            fut.set_result(res)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(batch_worker())

@app.post("/rerank")
async def rerank(req: RerankInput):
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    await QUEUE.put({"req": req, "future": fut})
    return await fut
