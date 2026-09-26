# Hugging Face Model Scraping Summary

**Date:** 2026-09-26  
**Total Models Fetched:** ~14,500+ (across all tasks)

---

## Tasks Completed

### 1. Top 10K Models by Downloads (Basic Metadata)
**Output:** `hf_models_top10k.tsv` (2.65 MB)

- **Source:** `https://huggingface.co/api/models?limit=10000&full=false`
- **Fields:** modelId, pipeline_tag, author, likes, downloads, tags
- **Use case:** Quick overview, filtering by task/downloads

---

### 2. Popular Text/Multimodal Models (>1,000 Downloads) - Full Data
**Output:** `hf_popular_text_mm/` (2,239 JSON files)

- **Filter:** pipeline_tag in {text-generation, text-classification, fill-mask, token-classification, question-answering, summarization, translation, zero-shot-classification, feature-extraction, sentence-similarity, image-text-to-text, visual-question-answering, image-classification, any-to-any}
- **Threshold:** downloads > 1,000
- **Per-model data:** metadata, model card (README), config, file tree, cardData, transformersInfo
- **Status:** Completed (~2,239 models)

---

### 3. Major Provider Deep Dive - Full Data
**Output:** `hf_provider_complete/` (2,226 JSON files + 8 index files)

| Provider | Organization | Models Fetched | Notes |
|----------|--------------|----------------|-------|
| **Gemma** | google | 500 | All Gemma variants |
| **Phi + more** | microsoft | 500 | Phi, DeBERTa, Florence, Table-Transformer, etc. |
| **Nemotron/Cosmos** | nvidia | 500 | Nemotron-3, Nemotron-4, Cosmos, GR00T, etc. |
| **Qwen** | Qwen | 468 | **All 468 models** from Qwen org |
| **DeepSeek** | deepseek-ai | 105 | V3, R1, Coder, Math, VL |
| **Mistral** | mistralai | 75 | Mistral, Mixtral, Ministral, Pixtral, Codestral |
| **Llama** | meta-llama | 70 | Llama 2, 3.1, 3.2, 3.3, Prompt Guard |
| **Cohere** | CohereForAI | 0 | No models under this org |

**Per-model data includes:**
- Complete metadata (tags, pipeline_tag, likes, downloads, private, gated, timestamps)
- **Model Card** (full README markdown content)
- **Config** (config.json - architecture, hidden_size, vocab_size, num_layers, etc.)
- **File Tree** (all repository files with sizes and LFS pointers)
- **CardData** (structured YAML frontmatter: license, datasets, metrics, eval results)
- **Transformers Info** (AutoModel class, architectures, pipeline tags)

**Index files:** `{provider}_index.json` - lightweight summaries for quick filtering

---

## Data Structure (Per Model JSON)

```json
{
  "modelId": "Qwen/Qwen2.5-7B-Instruct",
  "author": "Qwen",
  "sha": "abc123...",
  "lastModified": "2024-09-18T05:05:55.000Z",
  "tags": ["transformers", "safetensors", "qwen2", "text-generation", ...],
  "pipeline_tag": "text-generation",
  "library_name": "transformers",
  "likes": 15000,
  "downloads": 500000,
  "private": false,
  "gated": false,
  "disabled": false,
  "cardData": {
    "license": "apache-2.0",
    "datasets": ["Qwen/Qwen2.5-7B"],
    "metrics": [...],
    "model-index": [...]
  },
  "config": {
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 3584,
    "num_attention_heads": 28,
    "num_hidden_layers": 28,
    "vocab_size": 151936,
    ...
  },
  "siblings": [
    {"rfilename": "config.json", "size": 1234, "lfs": false},
    {"rfilename": "model.safetensors", "size": 14000000000, "lfs": true},
    ...
  ],
  "cardContent": "# Qwen2.5-7B-Instruct\n\n...",
  "cardDataParsed": {...},
  "transformersInfo": {
    "auto_model": "AutoModelForCausalLM",
    "pipeline_tag": "text-generation",
    "architectures": ["Qwen2ForCausalLM"]
  }
}
```

---

## Quick Query Examples

### Find all classification models from Qwen
```bash
grep "text-classification" hf_provider_complete/Qwen/*.json | jq -r .modelId
```

### Find all models with >100K downloads
```bash
jq -r 'select(.downloads > 100000) | .modelId' hf_provider_complete/*_index.json
```

### Find all Gemma models with config
```bash
ls hf_provider_complete/google--gemma*.json
```

### Load index for quick filtering
```python
import json
with open('hf_provider_complete/Qwen_index.json') as f:
    qwen_models = json.load(f)

# Filter by size tag
small_models = [m for m in qwen_models if any('0.5b' in t.lower() or '1.5b' in t.lower() for t in m['tags'])]
```

---

## Files Overview

```
C:\Users\manis\
├── hf_models_top10k.tsv              # 10K models, basic metadata (TSV)
├── hf_popular_text_mm/                # 2,239 models >1K downloads (full data)
│   ├── model1.json
│   ├── model2.json
│   └── ...
└── hf_provider_complete/              # 2,226 models from 8 providers (full data)
    ├── google--gemma-2-9b-it.json
    ├── microsoft--Phi-3.5-mini-instruct.json
    ├── nvidia--Nemotron-3-Nano-30B-A3B-NVFP4.json
    ├── Qwen--Qwen2.5-7B-Instruct.json
    ├── deepseek-ai--DeepSeek-V3.json
    ├── mistralai--Mistral-7B-Instruct-v0.2.json
    ├── meta-llama--Llama-3.1-8B-Instruct.json
    ├── google_index.json
    ├── microsoft_index.json
    ├── nvidia_index.json
    ├── Qwen_index.json
    ├── deepseek-ai_index.json
    ├── mistralai_index.json
    ├── meta-llama_index.json
    └── CohereForAI_index.json
```

---

## Tools Used

- **huggingface_hub** Python library (with auth token for higher rate limits)
- **HfApi.list_models()** - listing models with filters
- **HfApi.model_info()** - full model metadata + file tree
- **ModelCard.load()** - README content
- **Rate limiting:** ~0.03s between requests (authenticated)

---

## Next Steps / Ideas

1. **Query the Parquet dump** for full 1M+ model analysis: `huggingface/models-daily-dump`
2. **Build a local search index** (SQLite/DuckDB) from the JSON files
3. **Extract benchmark scores** from model cards for comparison tables
4. **Filter by size** (<32B) and task (classification) for your use case
5. **Download specific model weights** using `hf_hub_download()` when needed

---

*Generated automatically via huggingface_hub API*