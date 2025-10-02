import pandas as pd
import json

input_path = "chunk_ranking_kaggle_dev.jsonl"
output_path = "qrel_summary.csv"

# prepare an empty list to collect processed rows
results = []

# read in chunks (example: 500 rows at a time)
chunksize = 500
reader = pd.read_json(input_path, lines=True, chunksize=chunksize)
i = 0
for chunk in reader:
    print(f"Processing chunk {i}")
    i += 1
    for _, row in chunk.iterrows():
        qrel = row.get("qrel", {})
        if isinstance(qrel, str):
            try:
                qrel = json.loads(qrel)
            except:
                qrel = {}

        counts = {0: 0, 1: 0, 2: 0}
        for v in qrel.values():
            if v in counts:
                counts[v] += 1

        # extract question from messages
        question = ""
        messages = row.get("messages", [])
        if isinstance(messages, list):
            for m in messages:
                if m.get("role") == "user":
                    question = m.get("content", "").split("Question:")[-1].split("\n")[0].strip()
                    break

        results.append({
            "uuid": row.get("uuid"),
            "question": question,
            "qrel_0": counts[0],
            "qrel_1": counts[1],
            "qrel_2": counts[2]
        })

# convert results to DataFrame
df_summary = pd.DataFrame(results)

# save to CSV
df_summary.to_csv(output_path, index=False)

print("✅ Processed and saved summary to", output_path)
print(df_summary.head())
