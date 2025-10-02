import os
import json
import pandas as pd
from tqdm import tqdm
import tiktoken
from time import time
import re
# 🔹 Choose tokenizer (use OpenAI cl100k_base for GPT-3.5/4 tokenization)
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text, encoding=enc):
    if not isinstance(text, str):
        return 0
    return len(encoding.encode(text))




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


def process_row_for_tokens(row, row_idx):
    """
    Extract question, chunks, and count tokens for each.
    Returns list of dicts (one per chunk).
    """
    results = []

    # Parse messages
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

    # Collect chunks with token counts
    for ch in chunks_text:
        chunk_id = ch['chunk_id']
        chunk_content = ch['content']
        results.append({
            "uuid": row["uuid"],
            "row_idx": row_idx,
            "chunk_id": chunk_id,
            "question": question,
            "question_tokens": count_tokens(question),
            # "chunk_content": chunk_content,
            "chunk_tokens": count_tokens(clean_chunk_md(chunk_content))
        })

    return results

def main(jsonl_path, save_csv="token_counts.csv", chunksize=1000):
    reader = pd.read_json(jsonl_path, lines=True, chunksize=chunksize)
    all_records = []
    j=0
   
    for df_chunk in tqdm(reader, desc="Processing JSONL for token counts"):
        j+=1
        start_time = time()
        for i, row in df_chunk.iterrows():
            try:
                chunk_records = process_row_for_tokens(row, i)
                all_records.extend(chunk_records)
            except Exception as e:
                print(f"⚠️ Error at row {i}: {e}")
        elapsed = time() - start_time
        print(f"✅ Processed chunk {j} with {len(df_chunk)} rows in {elapsed:.2f} seconds")

    # Save to CSV
    df_out = pd.DataFrame(all_records)
    df_out.to_csv(save_csv, index=False)
    print(f"✅ Saved token counts to {save_csv}")

if __name__ == "__main__":
    main("chunk_ranking_kaggle_dev.jsonl", save_csv="question_chunk_token_counts.csv", chunksize=500)
