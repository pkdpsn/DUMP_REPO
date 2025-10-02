import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from time import time

# 🔹 Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔹 Create save directory
os.makedirs("embeddings", exist_ok=True)


def process_row(row, row_idx):
    """
    Extract question, chunks, and relevance scores from a single row.
    Returns a list of dicts: one per chunk.
    """
    results = []

    # Parse messages column if it's a string
    messages = row['messages']
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except:
            pass  # keep as is if already list/dict

    # Extract user question
    user_msg = None
    if isinstance(messages, list):
        user_msg = next((m for m in messages if m.get('role') == 'user'), None)
    elif isinstance(messages, dict) and messages.get('role') == 'user':
        user_msg = messages

    question = ""
    chunks_text = []
    if user_msg:
        content = user_msg.get('content', '')
        if "Question:" in content:
            parts = content.split("Text chunks:")
            question = parts[0].replace(
                "Identify the 10 most relevant text chunks for answering this question, then rank them in order of relevance (best first).",
                ""
            ).strip()
            if len(parts) > 1:
                chunks_raw = parts[1].split("[Chunk Index ")
                for ch in chunks_raw[1:]:
                    idx_end = ch.find("]")
                    idx = ch[:idx_end].strip()
                    text = ch[idx_end+1:].strip()
                    chunks_text.append({'chunk_id': idx, 'content': text})

    # Parse qrel column
    qrel_dict = row.get('qrel', {})
    if isinstance(qrel_dict, str):
        try:
            qrel_dict = json.loads(qrel_dict)
        except:
            qrel_dict = {}

    # Collect chunks
    for ch in chunks_text:
        chunk_id = ch['chunk_id']
        results.append({
            "id": f"{row['uuid']}_{row_idx}",
            "uuid": row["uuid"],
            "question": question,
            "chunk_id": chunk_id,
            "content": ch['content'],
            "qrel": qrel_dict.get(chunk_id, 0)
        })

    return results


def save_query_chunks_embeddings(chunks, query_uuid, model=embed_model, save_dir="embeddings"):
    """
    Save embeddings + metadata for all chunks of a query.
    """
    if len(chunks) == 0:
        return

    question_text = chunks[0]["question"]
    chunk_texts = [ch['content'] for ch in chunks]
    idx = chunks[0]["id"]

    # 🔹 Generate embeddings
    embeddings = model.encode(chunk_texts, convert_to_numpy=True)

    # 🔹 Save .npy
    npy_path = os.path.join(save_dir, f"{idx}_chunks.npy")
    np.save(npy_path, embeddings)

    # 🔹 Save metadata
    metadata = []
    for ch in chunks:
        meta = {
            "name": f"index_{query_uuid}_{ch['chunk_id']}_{ch['qrel']}",
            "uuid": query_uuid,
            "chunk_id": ch["chunk_id"],
            "qrel": ch["qrel"],
            "content": ch["content"],
            "question": question_text
        }
        metadata.append(meta)

    pkl_path = os.path.join(save_dir, f"{idx}_metadata.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ Saved {query_uuid}: {len(chunks)} chunks → {npy_path}, {pkl_path}")


def main(jsonl_path, chunksize=1000):
    reader = pd.read_json(jsonl_path, lines=True, chunksize=chunksize)
    j = 0
    for df_chunk in tqdm(reader, desc="Processing JSONL"):
        start_time = time()
        for i, row in df_chunk.iterrows():
            try:
                chunk_records = process_row(row, i)
                if not chunk_records:
                    print(f"⚠️ Error in row {i}")
                    continue
                query_uuid = row["uuid"]

                # 🔹 Save embeddings and metadata
                save_query_chunks_embeddings(chunk_records, query_uuid)

            except Exception as e:
                print(f"⚠️ Error at row {i}: {e}")

        elapsed = time() - start_time
        j += 1
        print(f"✅ Processed chunk {j} with {len(df_chunk)} rows in {elapsed:.2f} seconds")


if __name__ == "__main__":
    # 🔹 Change path to your dataset
    main("chunk_ranking_kaggle_dev.jsonl", chunksize=500)
