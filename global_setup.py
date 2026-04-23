import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# -----------------------------
# MODEL INIT
# -----------------------------
model_embed = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# FLATTEN JSON → STRUCTURED CHUNKS
# -----------------------------
def flatten_json(data):
    chunks = []

    def recurse(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_path = f"{path}.{k}" if path else k
                recurse(v, new_path)

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]"
                recurse(item, new_path)

        else:
            chunks.append({
                "text": f"{path}: {obj}",
                "path": path
            })

    recurse(data)
    return chunks


# -----------------------------
# REMOVE DUPLICATES
# -----------------------------
def deduplicate_chunks(chunks):
    seen = set()
    unique = []

    for c in chunks:
        if c["text"] not in seen:
            seen.add(c["text"])
            unique.append(c)

    return unique


# -----------------------------
# LOAD + EMBEDDING PRECOMPUTE
# -----------------------------
def load_memory_and_precompute(
    memory_path="veronica_memory.json",
    emb_path="chunk_embs.npy",
    force_recompute=False
):
    mem_file = Path(memory_path)

    if not mem_file.exists():
        raise FileNotFoundError(f"{memory_path} not found")

    data = json.loads(mem_file.read_text(encoding="utf-8"))

    # Step 1: Flatten
    chunks = flatten_json(data)

    # Step 2: Deduplicate
    chunks = deduplicate_chunks(chunks)

    emb_file = Path(emb_path)

    # Step 3: Load or compute embeddings
    if emb_file.exists() and not force_recompute:
        chunk_embs = np.load(emb_path)
    else:
        texts = [c["text"] for c in chunks]

        chunk_embs = model_embed.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        np.save(emb_path, chunk_embs)

    return chunks, chunk_embs


# -----------------------------
# SEARCH FUNCTION (CORE RETRIEVAL)
# -----------------------------
def search_memory(query, chunks, chunk_embs, top_k=5):
    query_emb = model_embed.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    # cosine similarity (dot product because normalized)
    scores = np.dot(chunk_embs, query_emb)

    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = []
    for i in top_indices:
        results.append({
            "text": chunks[i]["text"],
            "path": chunks[i]["path"],
            "score": float(scores[i])
        })

    return results


# -----------------------------
# STARTUP LOAD
# -----------------------------
CHUNKS, CHUNK_EMBS = load_memory_and_precompute()
