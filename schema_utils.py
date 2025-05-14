import json
import os
import sqlite3

SPIDER_PATH = "spider/database"

def load_schema_chunks(db_id, db_path):
    schema_file = os.path.join("spider", "tables.json")
    with open(schema_file, "r") as f:
        schemas = json.load(f)

    db_schema = next((s for s in schemas if s["db_id"] == db_id), None)
    if not db_schema:
        raise ValueError(f"Database schema '{db_id}' not found.")

    chunks = []
    table_map = {i: name for i, name in enumerate(db_schema["table_names_original"])}
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for table_idx, table_name in table_map.items():
        columns = [col[1] for i, col in enumerate(db_schema["column_names_original"]) if col[0] == table_idx]
        chunk = f"Table: {table_name}\nColumns: {', '.join(columns)}"

        # Try to add sample rows
        try:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 2")
            rows = cursor.fetchall()
            if rows:
                row_strs = ["- " + ", ".join(map(str, row)) for row in rows]
                chunk += "\nSample rows:\n" + "\n".join(row_strs)
        except Exception as e:
            print(f"⚠️ Skipping samples for table '{table_name}': {e}")

        chunks.append(chunk)

    conn.close()

    # Foreign key chunks
    for fk_pair in db_schema.get("foreign_keys", []):
        col1 = db_schema["column_names_original"][fk_pair[0]]
        col2 = db_schema["column_names_original"][fk_pair[1]]
        fk_chunk = f"Foreign Key: {col1[1]} → {col2[1]}"
        chunks.append(fk_chunk)

    return chunks

def list_databases(path=SPIDER_PATH):
    return sorted([
        name for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
    ])
