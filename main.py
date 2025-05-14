import sys
import os
import sqlite3
import re
import time
import torch
from schema_utils import load_schema_chunks, list_databases
from vector_store import RAGRetriever
from model_runner import run_gpt35, convert_sql_to_answer
from evaluator import evaluate_sql_outputs
from sqlparse import format as format_sql
from description_utils import load_descriptions, enrich_schema_with_descriptions
from logger import init_csv_log, log_result
from sentence_transformers import SentenceTransformer, util
os.environ["TOKENIZERS_PARALLELISM"] = "false"

init_csv_log()
SPIDER_PATH = "spider/database"
EMBEDDING_DIR = "schema_embeddings"
model = SentenceTransformer("all-MiniLM-L6-v2")
def extract_sql(gpt_output):
    # Try to extract SQL block from markdown-style formatting
    if "```sql" in gpt_output.lower():
        try:
            sql = gpt_output.lower().split("```sql")[1].split("```")[0].strip()
        except IndexError:
            sql = gpt_output.strip()
    else:
        sql = gpt_output.strip()

    # Attempt to find the first SELECT statement
    match = re.search(r"(select .*?)(;|$)", sql, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


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

from sentence_transformers import SentenceTransformer

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
You are an expert in writing SQLite-compatible SQL queries.

Your task is to generate a query that answers the user's question based on the provided database schema.

Rules:
- Use only the tables and fields found in the schema.
- Use JOINs only when necessary.
- If filtering by a text field (like name, title, city, or country), prefer flexible matches like:
    - `LIKE '%value%'`
    - or `LOWER(column) LIKE '%value%'` for case-insensitive search
- Do NOT hallucinate column or table names.
- Keep the query concise and focused.

Schema:
{schema_text}

Question:
{user_question}

SQL:
""".strip()

        gpt_output, token_usage = run_gpt35(rag_prompt)
        print("\n🧠 GPT Output:\n", gpt_output)

        evaluate_sql_outputs(user_question, gpt_output, verbose=False)
        sql_query = extract_sql(gpt_output)
        print("\n🧪 Extracted SQL:\n", sql_query)

        if not sql_query:
            print("⚠️ No SQL generated")
            return "SQL generation failed", "(⚠️ GPT failed to generate a valid SELECT query.)"

        rows = execute_sql_query(db_path, sql_query)
        answer = convert_sql_to_answer(rows, sql_query)
        print(f"\n🧾 Raw answer type: {type(answer)} — value: {answer}")

        # Normalize to string always
        if not answer:
            answer = "(No result returned)"
        elif not isinstance(answer, str):
            answer = str(answer)

        log_result(db_id, user_question, sql_query, answer, tokens=token_usage)
        return sql_query, answer

    except Exception as e:
        print(f"[ERROR] Exception in run_query: {e}")
        log_result(db_id, user_question, "", "", str(e))
        return "SQL generation failed", f"[Error] {str(e)}"


if __name__ == "__main__":
    try:
        user_question = input("Enter your question: ")
        t0 = time.time()


        question_embedding = model.encode(user_question, convert_to_tensor=True)

        if len(sys.argv) == 1:
            relevance_scores = []

            for db_id in list_databases():
                embedding_path = os.path.join(EMBEDDING_DIR, f"{db_id}.pt")
                if not os.path.exists(embedding_path):
                    continue
                try:
                    schema_embedding = torch.load(embedding_path)
                    score = util.pytorch_cos_sim(question_embedding, schema_embedding).item()
                    relevance_scores.append((score, db_id))
                except Exception as e:
                    print(f"⚠️ Skipping {db_id}: {e}")

            top_dbs = sorted(relevance_scores, reverse=True)[:3]
            

            for _, db_id in top_dbs:
                run_query(db_id, user_question)

        else:
            db_id = sys.argv[1]
            run_query(db_id, user_question)

        print(f"\n⏱️ Total time taken: {time.time() - t0:.2f} seconds")

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Exiting.")
