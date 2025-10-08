import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import re
import tiktoken

# ----------------- CONFIG -----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "colbert-ir/colbertv2.0"  # or any BERT model
MAX_TOKENS = 512
TOP_K = 10

# ----------------- SETUP -----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME , trust_remote_code=True).to(DEVICE)
model.eval()

MODEL_DIM = model.config.hidden_size
print(MODEL_DIM)

enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    if not isinstance(text, str):
        return 0
    return len(enc.encode(text))

def clean_text(text):
    """Remove HTML, markdown tables, extra spaces."""
    text = re.sub(r'<.*?>', ' ', text)
    text = '\n'.join([line for line in text.splitlines() if '|' not in line])
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s.,;:?!()-]', '', text)
    return text

# ----------------- DATA PREPROCESS -----------------
def extract_query_chunks(row):
    """Return list of dicts: each chunk with uuid, question, content, qrel, tokens."""
    results = []
    messages = row['messages']
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except:
            pass

    if isinstance(messages, list):
        user_msg = next((m for m in messages if m.get('role') == 'user'), None)
    elif isinstance(messages, dict) and messages.get('role') == 'user':
        user_msg = messages
    else:
        user_msg = None

    if not user_msg:
        return results

    content = user_msg.get('content', "")
    question = ""
    chunks_text = []

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

    qrel_dict = row.get("qrel", {}) if isinstance(row.get("qrel", {}), dict) else {}

    for ch in chunks_text:
        chunk_id = ch['chunk_id']
        results.append({
            "uuid": row["uuid"],
            "question": question,
            "chunk_id": chunk_id,
            "content": ch['content'],
            "qrel": qrel_dict.get(str(chunk_id), 0),
            "question_tokens": count_tokens(question),
            "chunk_tokens": count_tokens(clean_text(ch['content']))
        })

    return results

# ----------------- EMBEDDINGS -----------------
def get_token_embeddings_long(text, max_len=MAX_TOKENS):
    """Split long text into chunks and return concatenated token embeddings."""
    words = text.split()
    embeddings = []

    for i in range(0, len(words), max_len):
        chunk_text = " ".join(words[i:i+max_len])
        inputs = tokenizer(
            chunk_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_dim)
        attn_mask = inputs["attention_mask"].squeeze(0).bool()
        hidden = hidden[attn_mask]
        embeddings.append(hidden.cpu())
    if len(embeddings) == 0:
        return torch.zeros(1, MODEL_DIM)#.to(DEVICE)
    return torch.cat(embeddings, dim=0)

def cosine_sim(a, b):
    """Compute cosine similarity between 2 tensors: (N, dim) x (M, dim)."""
    a_norm = a / a.norm(dim=1, keepdim=True)
    b_norm = b / b.norm(dim=1, keepdim=True)
    return torch.mm(a_norm, b_norm.t())

# ----------------- RETRIEVAL -----------------
# def retrieve_topk_with_scores(question, chunks, topk=TOP_K):
#     """Modified to return scores along with chunks"""
#     q_emb = get_token_embeddings_long(question)  # (q_len, dim)
#     scores = []
#     i = 0
#     for ch in chunks:
#         i += 1
#         ch_emb = get_token_embeddings_long(ch['content'])  # (c_len, dim)
#         sim_matrix = cosine_sim(q_emb, ch_emb)  # late interaction
#         score = sim_matrix.max().item()  # ColBERT: max over token interactions
#         scores.append(score)

#     # rank
#     num_chunks = i
#     top_k = min(200, num_chunks // 3)
    
#     ranked_idx = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
#     topk_chunks = [(chunks[i], scores[i]) for i in ranked_idx[:topk]]
#     return topk_chunks

def retrieve_topk_with_scores(question, chunks, topk=TOP_K):
    """Modified to return scores along with chunks using correct ColBERT late interaction"""
    q_emb = get_token_embeddings_long(question)  # (q_len, dim)
    scores = []
    
    for ch in chunks:
        ch_emb = get_token_embeddings_long(ch['content'])  # (c_len, dim)
        sim_matrix = cosine_sim(q_emb, ch_emb)  # (q_len, c_len)
        
        # ColBERT late interaction: for each query token, find max similarity with chunk tokens
        # then sum across all query tokens
        max_sims = sim_matrix.max(dim=1)[0]  # max similarity for each query token
        score = max_sims.sum().item()  # sum of max similarities
        scores.append(score)

    # rank
    ranked_idx = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    topk_chunks = [(chunks[i], scores[i]) for i in ranked_idx[:topk]]
    return topk_chunks

def retrieve_topk(question, chunks, topk=TOP_K):
    """Original function for backward compatibility"""
    topk_with_scores = retrieve_topk_with_scores(question, chunks, topk)
    return [chunk for chunk, score in topk_with_scores]

# ----------------- EVALUATION -----------------
def evaluate(jsonl_path, topk=TOP_K):
    reader = pd.read_json(jsonl_path, lines=True)
    results = []
    detailed_results = []  # New: store detailed chunk-level results

    for i, row in tqdm(reader.iterrows(), total=len(reader), desc="Evaluating"):
        chunks = extract_query_chunks(row)
        if not chunks:
            continue
        question = chunks[0]["question"]
        total_chunks = len(chunks)  # total number of chunks for this query

        # Get chunks with scores
        topk_res_with_scores = retrieve_topk_with_scores(question, chunks, topk=topk)
        topk_res = [chunk for chunk, score in topk_res_with_scores]

        # Count qrel 1 and 2
        all_rel_1 = sum([1 for c in chunks if c["qrel"] == 1])
        all_rel_2 = sum([1 for c in chunks if c["qrel"] == 2])
        retrieved_rel_1 = sum([1 for c in topk_res if c["qrel"] == 1])
        retrieved_rel_2 = sum([1 for c in topk_res if c["qrel"] == 2])

        results.append({
            "uuid": row["uuid"],
            "question": question,
            "total_chunks": total_chunks,
            "total_rel_1": all_rel_1,
            "total_rel_2": all_rel_2,
            "retrieved_rel_1": retrieved_rel_1,
            "retrieved_rel_2": retrieved_rel_2,
            "missing_rel_1": all_rel_1 - retrieved_rel_1,
            "missing_rel_2": all_rel_2 - retrieved_rel_2
        })

        # Store detailed chunk-level results
        for idx, (chunk, score) in enumerate(topk_res_with_scores):
            detailed_results.append({
                "uuid": row["uuid"],
                "uuid_row": f"{row['uuid']}_{i}",
                "chunk_number": chunk["chunk_id"],
                "colbert_score": score,
                "qrel": chunk["qrel"]
            })

        print(f"Q {row['uuid']}: Total chunks {total_chunks}, "
              f"R1 {retrieved_rel_1}/{all_rel_1}, R2 {retrieved_rel_2}/{all_rel_2}")
        if i >= 3:  
            break

    return results, detailed_results

# ----------------- MAIN -----------------
if __name__ == "__main__":
    dataset_path = "shuffled_15_rows.jsonl"  # path to your JSONL file
    topk = 100 #### uselesss

    eval_results, detailed_results = evaluate(dataset_path, topk=topk)

    # summary
    r1_tot = sum(r["total_rel_1"] for r in eval_results)
    r2_tot = sum(r["total_rel_2"] for r in eval_results)
    r1_retr = sum(r["retrieved_rel_1"] for r in eval_results)
    r2_retr = sum(r["retrieved_rel_2"] for r in eval_results)
    print("\n==== Summary ====")
    print(f"R1 Total: {r1_tot}, Retrieved: {r1_retr}, Missing: {r1_tot-r1_retr}")
    print(f"R2 Total: {r2_tot}, Retrieved: {r2_retr}, Missing: {r2_tot-r2_retr}")

    # Save detailed results to CSV
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv("colbert_detailed_results.csv", index=False)
    print(f" Detailed results saved to 'colbert_detailed_results.csv'")