import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model (same one you used for chunks)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def evaluate_retrieval(embeddings_dir="embeddings", top_k=10):
    results = []
    files = [f for f in os.listdir(embeddings_dir) if f.endswith("_metadata.pkl")]

    for meta_file in files:
        base = meta_file.replace("_metadata.pkl", "")
        npy_file = os.path.join(embeddings_dir, base + "_chunks.npy")
        pkl_file = os.path.join(embeddings_dir, meta_file)

        # Load data
        with open(pkl_file, "rb") as f:
            metadata = pickle.load(f)
        chunk_embeds = np.load(npy_file)

        # Encode query
        question = metadata[0]["question"]
        q_emb = embed_model.encode([question], convert_to_numpy=True)
        if q_emb.shape[1]!= chunk_embeds.shape[1]:
            print("skiping")
            continue

        # Compute cosine similarities
        sims = cosine_similarity(q_emb, chunk_embeds)[0]
        ranked_idx = np.argsort(sims)[::-1]  # descending

        # Get top-k chunks
        topk_idx = ranked_idx[:top_k]
        retrieved_chunks = [metadata[i] for i in topk_idx]

        # Count relevant chunks by qrel level
        all_rel_1 = [m for m in metadata if m["qrel"] == 1]
        all_rel_2 = [m for m in metadata if m["qrel"] == 2]
        retrieved_rel_1 = [m for m in retrieved_chunks if m["qrel"] == 1]
        retrieved_rel_2 = [m for m in retrieved_chunks if m["qrel"] == 2]

        missing_1 = len(all_rel_1) - len(retrieved_rel_1)
        missing_2 = len(all_rel_2) - len(retrieved_rel_2)

        results.append({
            "uuid": metadata[0]["uuid"],
            "question": question,
            "total_rel_1": len(all_rel_1),
            "total_rel_2": len(all_rel_2),
            "retrieved_rel_1": len(retrieved_rel_1),
            "retrieved_rel_2": len(retrieved_rel_2),
            "missing_rel_1": missing_1,
            "missing_rel_2": missing_2
        })

        print(f"Evaluated {metadata[0]['uuid']}: "
              f"R1 {len(retrieved_rel_1)}/{len(all_rel_1)}, "
              f"R2 {len(retrieved_rel_2)}/{len(all_rel_2)}")

    return results


if __name__ == "__main__":
    eval_results = evaluate_retrieval(embeddings_dir="embeddings", top_k=100)

    # Show summary
    r1_tot = 0 
    r2_tot = 0
    r1_miss = 0
    r2_miss = 0
    
    for r in eval_results:
        print(f"Q ({r['uuid']}): {r['question'][:60]}...")
        print(f"   Rel-1: {r['retrieved_rel_1']}/{r['total_rel_1']} "
              f"(Missing {r['missing_rel_1']}) | "
              f"Rel-2: {r['retrieved_rel_2']}/{r['total_rel_2']} "
              f"(Missing {r['missing_rel_2']})\n")
        r1_tot += r["total_rel_1"]
        r2_tot += r["total_rel_2"]
        r1_miss += r["missing_rel_1"]
        r2_miss += r["missing_rel_2"]
        
    print(f"R1 Total: {r1_tot}, Retrieved: {r1_tot - r1_miss}, Missing: {r1_miss}")
    print(f"R2 Total: {r2_tot}, Retrieved: {r2_tot - r2_miss}, Missing: {r2_miss}")
