import json
import os

def load_schema_chunks(db_id, db_path):
    schema_file = os.path.join("spider", "tables.json")
    with open(schema_file, "r") as f:
        schemas = json.load(f)

    db_schema = next((s for s in schemas if s["db_id"] == db_id), None)
    if not db_schema:
        raise ValueError(f"Database schema '{db_id}' not found.")

    chunks = []
    for table_name, table_idx in zip(db_schema["table_names_original"], range(len(db_schema["table_names_original"]))):
        columns = [col[1] for i, col in enumerate(db_schema["column_names_original"]) if col[0] == table_idx]
        chunk = f"Table: {table_name}\nColumns: {', '.join(columns)}\n"
        chunks.append(chunk)

    for fk_pair in db_schema.get("foreign_keys", []):
        col1 = db_schema["column_names_original"][fk_pair[0]]
        col2 = db_schema["column_names_original"][fk_pair[1]]
        fk_chunk = f"Foreign Key: {col1[1]} → {col2[1]}\n"
        chunks.append(fk_chunk)

    return chunks
SPIDER_PATH = "spider/database"
def list_databases(path=SPIDER_PATH):
    return sorted([
        name for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
    ])