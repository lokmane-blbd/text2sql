import json

def load_descriptions(path="descriptions.json"):
    with open(path, "r") as f:
        return json.load(f)

def enrich_schema_with_descriptions(schema_chunks, db_id, descriptions):
    enriched = []
    db_info = descriptions.get(db_id, {})
    table_descs = db_info.get("tables", {})
    
    for chunk in schema_chunks:
        if "Table:" in chunk:
            lines = chunk.strip().split("\n")
            first_line = lines[0].strip()  # "Table: singer"
            table_name = first_line.replace("Table:", "").strip()

            desc = table_descs.get(table_name)
            if desc:
                lines.append(f"Description: {desc}")
            enriched.append("\n".join(lines))
        else:
            enriched.append(chunk)
    return enriched
