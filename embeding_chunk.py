import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
import torch
from transformers import AutoTokenizer, AutoModel

# ----------------------
# Device setup
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------
# Load FinBERT model
# ----------------------
MODEL_NAME = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# ----------------------
# Create save directory
# ----------------------
EMB_DIR = "embeddings"
os.makedirs(EMB_DIR, exist_ok=True)

# ----------------------
# Utilities
# ----------------------
def process_row(row, row_idx):
    """Extract question, chunks, and relevance scores from a single row."""
    results = []
    messages = row['messages']
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except:
            pass

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

    qrel_dict = row.get('qrel', {})
    if isinstance(qrel_dict, str):
        try:
            qrel_dict = json.loads(qrel_dict)
        except:
            qrel_dict = {}

    for ch in chunks_text:
        chunk_id = ch['chunk_id']
        results.append({
            "id": f"{row['uuid']}_{row_idx}_{chunk_id}",
            "uuid": row["uuid"],
            "question": question,
            "chunk_id": chunk_id,
            "content": ch['content'],
            "qrel": qrel_dict.get(chunk_id, 0)
        })

    return results

# ----------------------
# Embedding function
# ----------------------
def embed_texts(texts, pooling="cls", batch_size=8):
    embeddings = []
    max_length = 512  # BERT limitation

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = model(**tokens)
            if pooling == "cls":
                emb = outputs.last_hidden_state[:, 0, :]
            elif pooling == "mean":
                mask = tokens['attention_mask'].unsqueeze(-1)
                emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
            else:
                raise ValueError("pooling must be 'cls' or 'mean'")
        embeddings.append(emb.cpu().numpy())

    return np.vstack(embeddings)

# ----------------------
# Save embeddings + metadata
# ----------------------
def save_query_chunks_embeddings(chunks, query_uuid, save_dir=EMB_DIR, pooling="cls", batch_size=8):
    if not chunks:
        return

    question_text = chunks[0]["question"]
    chunk_texts = [ch['content'] for ch in chunks]
    idx = chunks[0]["id"]

    embeddings = embed_texts(chunk_texts, pooling=pooling, batch_size=batch_size)
    np.save(os.path.join(save_dir, f"{idx}_chunks.npy"), embeddings)

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

    with open(os.path.join(save_dir, f"{idx}_metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ Saved {query_uuid}: {len(chunks)} chunks → {idx}_chunks.npy")

# ----------------------
# Main processing loop
# ----------------------
def main(jsonl_path, chunksize=100, pooling="cls", batch_size=8):
    reader = pd.read_json(jsonl_path, lines=True, chunksize=chunksize)
    chunk_num = 0
    for df_chunk in tqdm(reader, desc="Processing JSONL"):
        start_time = time()
        for i, row in df_chunk.iterrows():
            try:
                chunk_records = process_row(row, i)
                if not chunk_records:
                    print(f"⚠️ Error parsing row {i}")
                    continue
                query_uuid = row["uuid"]
                save_query_chunks_embeddings(chunk_records, query_uuid, pooling=pooling, batch_size=batch_size)
            except Exception as e:
                print(f"⚠️ Error at row {i}: {e}")
        elapsed = time() - start_time
        chunk_num += 1
        print(f"✅ Processed chunk {chunk_num} with {len(df_chunk)} rows in {elapsed:.2f}s")

# ----------------------
# Run
# ----------------------
if __name__ == "__main__":
    main(
        "chunk_ranking_kaggle_dev.jsonl",
        chunksize=100,
        pooling="cls",
        batch_size=4
    )

# import os
# import json
# import pickle
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from time import time
# import torch
# from transformers import AutoTokenizer, AutoModel

# # 🔹 Device setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

# # 🔹 Load Longformer model
# model_name = "/finbert"  # you can switch to longfformer-large-4096 if needed
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name).to(device)
# model.eval()

# # 🔹 Create save directory
# os.makedirs("embeddings", exist_ok=True)


# def process_row(row, row_idx):
#     """Extract question, chunks, and relevance scores from a single row."""
#     results = []
#     messages = row['messages']
#     if isinstance(messages, str):
#         try:
#             messages = json.loads(messages)
#         except:
#             pass

#     user_msg = None
#     if isinstance(messages, list):
#         user_msg = next((m for m in messages if m.get('role') == 'user'), None)
#     elif isinstance(messages, dict) and messages.get('role') == 'user':
#         user_msg = messages

#     question = ""
#     chunks_text = []
#     if user_msg:
#         content = user_msg.get('content', '')
#         if "Question:" in content:
#             parts = content.split("Text chunks:")
#             question = parts[0].replace(
#                 "Identify the 10 most relevant text chunks for answering this question, then rank them in order of relevance (best first).",
#                 ""
#             ).strip()
#             if len(parts) > 1:
#                 chunks_raw = parts[1].split("[Chunk Index ")
#                 for ch in chunks_raw[1:]:
#                     idx_end = ch.find("]")
#                     idx = ch[:idx_end].strip()
#                     text = ch[idx_end+1:].strip()
#                     chunks_text.append({'chunk_id': idx, 'content': text})

#     qrel_dict = row.get('qrel', {})
#     if isinstance(qrel_dict, str):
#         try:
#             qrel_dict = json.loads(qrel_dict)
#         except:
#             qrel_dict = {}

#     for ch in chunks_text:
#         chunk_id = ch['chunk_id']
#         results.append({
#             "id": f"{row['uuid']}_{row_idx}",
#             "uuid": row["uuid"],
#             "question": question,
#             "chunk_id": chunk_id,
#             "content": ch['content'],
#             "qrel": qrel_dict.get(chunk_id, 0)
#         })

#     return results


# def embed_texts_longformer(texts, pooling="cls", batch_size=2):
#     """Encode a list of texts with Longformer and return embeddings."""
#     embeddings = []
#     for i in range(0, len(texts), batch_size):
#         batch = texts[i:i+batch_size]
#         tokens = tokenizer(
#             batch,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=4096  # Longformer max length
#         ).to(device)

#         # Longformer requires attention mask; by default tokenizer provides it
#         with torch.no_grad():
#             outputs = model(**tokens)
#             if pooling == "cls":
#                 emb = outputs.last_hidden_state[:, 0, :]  # CLS token
#             else:
#                 emb = outputs.last_hidden_state.mean(dim=1)  # mean pooling
#         embeddings.append(emb.cpu().numpy())

#     return np.vstack(embeddings)


# def save_query_chunks_embeddings(chunks, query_uuid, save_dir="embeddings", pooling="cls"):
#     """Save embeddings and metadata for all chunks of a query."""
#     if not chunks:
#         return

#     question_text = chunks[0]["question"]
#     chunk_texts = [ch['content'] for ch in chunks]
#     idx = chunks[0]["id"]

#     # 🔹 Embed chunks in batches
#     embeddings = embed_texts_longformer(chunk_texts, pooling=pooling)

#     # Save embeddings
#     npy_path = os.path.join(save_dir, f"{idx}_chunks.npy")
#     np.save(npy_path, embeddings)

#     # Save metadata
#     metadata = []
#     for ch in chunks:
#         meta = {
#             "name": f"index_{query_uuid}_{ch['chunk_id']}_{ch['qrel']}",
#             "uuid": query_uuid,
#             "chunk_id": ch["chunk_id"],
#             "qrel": ch["qrel"],
#             "content": ch["content"],
#             "question": question_text
#         }
#         metadata.append(meta)

#     pkl_path = os.path.join(save_dir, f"{idx}_metadata.pkl")
#     with open(pkl_path, "wb") as f:
#         pickle.dump(metadata, f)

#     print(f"✅ Saved {query_uuid}: {len(chunks)} chunks → {npy_path}, {pkl_path}")


# def main(jsonl_path, chunksize=1000, pooling="cls", batch_size=2):
#     reader = pd.read_json(jsonl_path, lines=True, chunksize=chunksize)
#     chunk_num = 0
#     for df_chunk in tqdm(reader, desc="Processing JSONL"):
#         start_time = time()
#         for i, row in df_chunk.iterrows():
#             try:
#                 chunk_records = process_row(row, i)
#                 if not chunk_records:
#                     print(f"⚠️ Error parsing row {i}")
#                     continue
#                 query_uuid = row["uuid"]
#                 save_query_chunks_embeddings(chunk_records, query_uuid, pooling=pooling)
#             except Exception as e:
#                 print(f"⚠️ Error at row {i}: {e}")
#         elapsed = time() - start_time
#         chunk_num += 1
#         print(f"✅ Processed chunk {chunk_num} with {len(df_chunk)} rows in {elapsed:.2f} seconds")


# if __name__ == "__main__":
#     # pooling="cls" or pooling="mean"
#     main("chunk_ranking_kaggle_dev.jsonl", chunksize=50, pooling="cls", batch_size=1)
