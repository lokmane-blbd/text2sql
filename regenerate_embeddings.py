# regenerate_embeddings.py

import os
import torch
from sentence_transformers import SentenceTransformer
from schema_utils import load_schema_chunks, list_databases

SPIDER_PATH = "spider/database"
EMBEDDING_DIR = "schema_embeddings"
model = SentenceTransformer("all-MiniLM-L6-v2")

os.makedirs(EMBEDDING_DIR, exist_ok=True)

def generate_embeddings_for_all():
    db_ids = list_databases()
    print(f"Found {len(db_ids)} databases...")

    for db_id in db_ids:
        db_path = os.path.join(SPIDER_PATH, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            print(f"⚠️ Skipping {db_id}: SQLite file not found.")
            continue

        print(f"📦 Processing {db_id}...")

        try:
            chunks = load_schema_chunks(db_id, db_path)
            chunk_text = " ".join(chunks).replace("\n", " ").strip()
            emb = model.encode(chunk_text, convert_to_tensor=True)

            torch.save(emb, os.path.join(EMBEDDING_DIR, f"{db_id}.pt"))
            print(f"✅ Saved embedding for {db_id}\n")

        except Exception as e:
            print(f"❌ Failed on {db_id}: {e}")

if __name__ == "__main__":
    generate_embeddings_for_all()
