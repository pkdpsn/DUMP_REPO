import os
import json
import pickle
import hashlib
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

import re

def clean_chunk_md(text):
    """
    Clean a chunk of text from Markdown, tables, HTML tags, and extra whitespace.
    """
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    
    # Remove Markdown tables (lines with |)
    text = '\n'.join([line for line in text.splitlines() if '|' not in line])
    
    # Remove extra newlines and spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Optional: remove non-alphanumeric symbols except punctuation
    text = re.sub(r'[^\w\s.,;:?!()-]', '', text)
    
    return text


# ----------------------
# Config
# ----------------------
MODEL_NAME = "yiyanghkust/finbert-tone"  # or colbert-ir/colbertv2.0
EMBEDDING_CACHE_DIR = "embedding_cache"
JSONL_PATH = "chunk_ranking_kaggle_dev.jsonl"
MAX_LENGTH = 512
BATCH_SIZE = 4  # adjust based on GPU memory

os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# ----------------------
# Utility Functions
# ----------------------
def get_cache_key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def save_embeddings(text: str, embeddings: torch.Tensor, attention_mask: torch.Tensor):
    key = get_cache_key(text)
    path = os.path.join(EMBEDDING_CACHE_DIR, f"{key}.pkl")
    with open(path, "wb") as f:
        pickle.dump({
            "embedding": embeddings.cpu().numpy(),
            "attention_mask": attention_mask.cpu().numpy()
        }, f)


def embed_texts(texts, batch_size=BATCH_SIZE, max_length=MAX_LENGTH):
    all_embeddings = []
    all_masks = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(batch,
                   padding="max_length",  # pad everything to max_length
                   truncation=True,
                   max_length=MAX_LENGTH,
                   return_tensors="pt")

        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = model(**tokens)
            # token-level embeddings: [batch, seq_len, hidden_dim]
            embeddings = outputs.last_hidden_state
        all_embeddings.append(embeddings.cpu())
        all_masks.append(tokens['attention_mask'].cpu())

    return torch.cat(all_embeddings, dim=0), torch.cat(all_masks, dim=0)


# ----------------------
# Dataset Iterator
# ----------------------
def process_row(row):
    """Extract question and chunk texts from one row."""
    messages = row["messages"]
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except:
            return "", []

    user_msg = None
    if isinstance(messages, list):
        user_msg = next((m for m in messages if m.get("role") == "user"), None)
    elif isinstance(messages, dict) and messages.get("role") == "user":
        user_msg = messages

    question = ""
    chunks = []
    if user_msg:
        content = user_msg.get("content", "")
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
                    text = ch[idx_end + 1:].strip()
                    chunks.append(clean_chunk_md(text))
    return question, chunks


# ----------------------
# Main Preprocessing
# ----------------------
def build_embedding_cache(jsonl_path, chunksize=500):
    reader = pd.read_json(jsonl_path, lines=True, chunksize=chunksize)
    total_q, total_c = 0, 0

    for df_chunk in tqdm(reader, desc="Embedding dataset"):
        for _, row in df_chunk.iterrows():
            try:
                question, chunks = process_row(row)
                # Embed question
                if question:
                    emb, mask = embed_texts([question])
                    save_embeddings(question, emb[0], mask[0])
                    total_q += 1
                # Embed chunks
                if chunks:
                    emb, mask = embed_texts(chunks)
                    for i, ch in enumerate(chunks):
                        save_embeddings(ch, emb[i], mask[i])
                        total_c += 1
            except Exception as e:
                print(f"⚠️ Error parsing row: {e}")
            finally:
                print(f"Processed {total_q} questions and {total_c} chunks so far.", end="\r")

    print(f"\n✅ Cached {total_q} questions and {total_c} chunks as embeddings.")


if __name__ == "__main__":
    build_embedding_cache(JSONL_PATH, chunksize=50)
