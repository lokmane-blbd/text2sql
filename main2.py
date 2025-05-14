import sys
import os
import sqlite3
import re
import time
from schema_utils import load_schema_chunks, list_databases
from vector_store import RAGRetriever
from model_runner import run_gpt35, convert_sql_to_answer
from evaluator import evaluate_sql_outputs
from sqlparse import format as format_sql
from description_utils import load_descriptions, enrich_schema_with_descriptions
from logger import init_csv_log, log_result
from sentence_transformers import SentenceTransformer, util

init_csv_log()
SPIDER_PATH = "spider/database"

def extract_sql(gpt_output):
    if "```sql" in gpt_output:
        sql = gpt_output.split("```sql")[1].split("```")[0].strip()
    else:
        sql = gpt_output.strip()

    if not sql.lower().startswith("select"):
        return None

    # Fix YEAR()
    sql = re.sub(r"YEAR\(([^)]+)\)", r"strftime('%Y', \1)", sql, flags=re.IGNORECASE)

    if "select" in sql.lower() and "distinct" not in sql.lower():
        sql = re.sub(r"(?i)^select", "SELECT DISTINCT", sql, count=1)

    return sql

def execute_sql_query(db_path, query):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        return f"[Execution Error] {e}"

def get_schema_score(question, enriched_chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode(question, convert_to_tensor=True)
    schema_text = " ".join(enriched_chunks)
    schema_embedding = model.encode(schema_text, convert_to_tensor=True)
    return util.pytorch_cos_sim(query_embedding, schema_embedding).item()

def run_query(db_id, user_question):
    db_path = f"{SPIDER_PATH}/{db_id}/{db_id}.sqlite"

    try:
        descriptions = load_descriptions()
        schema_chunks = load_schema_chunks(db_id, db_path)
        enriched_chunks = enrich_schema_with_descriptions(schema_chunks, db_id, descriptions)

        retriever = RAGRetriever(collection_name=f"schema_chunks_{db_id}")
        retriever.add_chunks(enriched_chunks)
        retrieved_chunks = retriever.retrieve(user_question, k=4)

        schema_text = " | ".join([
            chunk.replace("\n", " ").strip()
            for chunk in retrieved_chunks if "Table:" in chunk
        ])

        rag_prompt = f"""
You are an expert in SQL. Write an SQLite-compatible SQL query that answers the question below.
Use only the relevant tables based on the schema.
Always fully qualify column names with their table name (e.g., table.column).

Schema: {schema_text}

Question: {user_question}

SQL:
""".strip()

        gpt_output, token_usage = run_gpt35(rag_prompt)

        evaluate_sql_outputs(user_question, gpt_output, verbose=False)

        sql_query = extract_sql(gpt_output)

        if not sql_query:
            log_result(db_id, user_question, gpt_output, "", "Invalid SQL from GPT", tokens=token_usage)
            return

        rows = execute_sql_query(db_path, sql_query)
        answer = convert_sql_to_answer(rows, sql_query)

        log_result(db_id, user_question, sql_query, answer, tokens=token_usage)

        answer_str = str(answer).strip().lower()
        if answer and not any(kw in answer_str for kw in ["none", "empty", "[]", "no result", "the result is: none"]):
            print("\n\n\n")
            print(f"Answer: {answer}")

    except Exception as e:
        log_result(db_id, user_question, "", "", str(e))

if __name__ == "__main__":
    try:
        user_question = input("Enter your question: ")
        t2 = time.time()

        relevance_scores = []

        for db_id in list_databases():
            try:
                db_path = f"{SPIDER_PATH}/{db_id}/{db_id}.sqlite"
                schema_chunks = load_schema_chunks(db_id, db_path)
                enriched_chunks = enrich_schema_with_descriptions(schema_chunks, db_id, load_descriptions())
                score = get_schema_score(user_question, enriched_chunks)
                relevance_scores.append((score, db_id))
            except Exception as e:
                print(f"⚠️ Skipping {db_id}: {e}")
                continue

        top_dbs = sorted(relevance_scores, reverse=True)[:5]
        print(f"\n🔎 Running on top {len(top_dbs)} databases based on relevance scores:")
        for score, db_id in top_dbs:
            print(f"  - {db_id}: {score:.4f}")
        for _, db_id in top_dbs:
            run_query(db_id, user_question)

        total_time = time.time() - t2
        print(f"\n⏱️ Total time taken: {total_time:.2f} seconds")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Exiting.")
