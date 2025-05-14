import os
import torch
from schema_utils import load_schema_chunks
from description_utils import load_descriptions, enrich_schema_with_descriptions
from sentence_transformers import SentenceTransformer

SPIDER_PATH = "spider/database"
EMBEDDING_DIR = "schema_embeddings"
os.makedirs(EMBEDDING_DIR, exist_ok=True)

def list_databases(path=SPIDER_PATH):
    return sorted([
        name for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
    ])

def compute_and_save_embeddings():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    descriptions = load_descriptions()

    for db_id in list_databases():
        try:
            db_path = f"{SPIDER_PATH}/{db_id}/{db_id}.sqlite"  # can be dummy if .sqlite is missing
            schema_chunks = load_schema_chunks(db_id, db_path)
            enriched_chunks = enrich_schema_with_descriptions(schema_chunks, db_id, descriptions)

            schema_text = " ".join(enriched_chunks)  # combine all chunks as one text
            embedding = model.encode(schema_text, convert_to_tensor=True)

            torch.save(embedding, os.path.join(EMBEDDING_DIR, f"{db_id}.pt"))
            print(f"✅ Saved embedding for {db_id}")
        except Exception as e:
            print(f"❌ Failed to process {db_id}: {e}")

if __name__ == "__main__":
    compute_and_save_embeddings()
