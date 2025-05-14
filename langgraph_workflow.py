from langgraph.graph import StateGraph
from typing import TypedDict
from sentence_transformers import SentenceTransformer, util
from langchain_core.runnables import RunnableLambda
import torch
import os
import io
import sys
import json
from difflib import get_close_matches
from schema_utils import list_databases
from schema_utils import list_databases
from main import run_query
from model_runner import run_gpt35
from typing import Tuple
from langgraph.graph import StateGraph






EMBEDDING_DIR = "schema_embeddings"
model = SentenceTransformer("all-MiniLM-L6-v2")
class QueryState(TypedDict):
    question: str
    dbs: list[str]
    output: str
    attempt: int
    hinted_db: str | None
    hinted_column: str | None
    gpt_selected_dbs: list[str]
    all_outputs: dict[str, str]
    final_db: str | None          # ✅ New
    final_sql: str | None         # ✅ New



def resolve_fuzzy_db_name(name: str) -> str | None:
    all_dbs = list_databases()
    matches = get_close_matches(name.lower(), all_dbs, n=1, cutoff=0.6)
    return matches[0] if matches else None

def resolve_fuzzy_list(names: list[str]) -> list[str]:
    resolved = []
    for name in names:
        match = resolve_fuzzy_db_name(name)
        if match:
            resolved.append(match)
    return resolved

# === Node: Extract hints from user (NEW AGENT) ===
def extract_context(state: QueryState) -> QueryState:
    print("🕵️ extract_context")
    question = state["question"]

    prompt = f"""
From the following user input, extract the name of a database or column if mentioned.

Return a JSON object with:
- "hinted_db": the database name if clearly stated
- "hinted_column": the column name if clearly stated

User input:
\"\"\"{question}\"\"\"
""".strip()

    try:
        response, _ = run_gpt35(prompt)
        parsed = json.loads(response)
        return {**state, "hinted_db": parsed.get("hinted_db"), "hinted_column": parsed.get("hinted_column")}
    except Exception as e:
        print(f"⚠️ Failed to extract context: {e}")
        return {**state, "hinted_db": None, "hinted_column": None}

def format_with_gpt(raw_output: str, question: str) -> str:
    prompt = f"""
You are a helpful assistant.

Rewrite the database output into a clean, natural, and concise sentence that directly answers the question.

Rules:
- Make it user-friendly and easy to understand.
-DO NOT explain how it was found.
- Do not mention SQL, databases, or any internal logic.
- Keep it user-friendly.
- Preserve all factual data from the original output.
- Do not hallucinate or add extra assumptions.

Question:
\"\"\"{question}\"\"\"

Database result:
\"\"\"{raw_output.strip()}\"\"\"

Final answer:
""".strip()

    try:
        response, _ = run_gpt35(prompt)
        return response.strip()
    except Exception as e:
        print(f"⚠️ GPT formatting failed: {e}")
        return raw_output  # fallback if GPT fails



def retrieve_schema(state: QueryState) -> QueryState:
    print("🔍 retrieve_schema")
    question = state["question"]
    # Priority 1: use hinted_db if specified
    if state.get("hinted_db"):
        raw_hint = state["hinted_db"]
        matched = resolve_fuzzy_db_name(raw_hint)
        if matched:
            print(f"🎯 Hinted DB matched: {raw_hint} → {matched}")
            return {**state, "dbs": [matched]}
        else:
            print(f"❌ No DB match found for hinted: {raw_hint}")

    # Priority 2: GPT-selected DBs (clean them with fuzzy match)
    if state.get("gpt_selected_dbs"):
        fuzzy_cleaned = resolve_fuzzy_list(state["gpt_selected_dbs"])
        if fuzzy_cleaned:
            print(f"🧠 GPT-selected DBs (cleaned): {fuzzy_cleaned}")
            return {**state, "dbs": fuzzy_cleaned}
        else:
            print("⚠️ No valid DBs from GPT-selected list.")

    # Fallback: use embeddings
    question_embedding = model.encode(question, convert_to_tensor=True)
    relevance_scores = []

    for db_id in list_databases():
        emb_path = os.path.join(EMBEDDING_DIR, f"{db_id}.pt")
        if not os.path.exists(emb_path):
            continue
        try:
            schema_emb = torch.load(emb_path)
            score = util.pytorch_cos_sim(question_embedding, schema_emb).item()
            relevance_scores.append((score, db_id))
        except Exception as e:
            print(f"⚠️ Skipping {db_id}: {e}")

    top_dbs = [db_id for _, db_id in sorted(relevance_scores, reverse=True)[:3]]

    if not top_dbs:
        print("❌ No databases found via embedding fallback.")
        return {**state, "dbs": []}

    print(f"🧪 Embedding fallback selected DBs: {top_dbs}")
    return {**state, "dbs": top_dbs}



def select_best_answer(state: QueryState) -> QueryState:
    print("🧠 select_best_answer ()")

    question = state["question"]
    all_outputs = state.get("all_outputs", {})
    print("🧾 All outputs seen by GPT:")
    for db, out in all_outputs.items():
        print(f"--- {db} ---")
        print(out)

    formatted_outputs = json.dumps(all_outputs, indent=2)


    prompt = f"""
You are selecting the most relevant database output for the user's question.

Question:
\"\"\"{question}\"\"\"

Below is a JSON object where keys are database names and values are their respective outputs:

{formatted_outputs}

Pick the best database name key (e.g. "cinema") that gives the most complete and relevant answer. Return only the key.
"""

    try:
        response, _ = run_gpt35(prompt)
        selected_db_raw = response.strip().strip('"').strip("'").lower()

        # Fuzzy match GPT's selection to valid db keys in all_outputs
        possible_dbs = list(all_outputs.keys())
        matched = get_close_matches(selected_db_raw, possible_dbs, n=1, cutoff=0.6)
        
        if matched:
            selected_db = matched[0]
            print(f"✅ GPT selected DB (matched): {selected_db}")
            selected_output = all_outputs[selected_db].strip()
            import re
            sql_match = re.search(r"📝 SQL used:\n(.+)", selected_output, re.DOTALL)
            extracted_sql = sql_match.group(1).strip() if sql_match else "SQL not found"
            return {
            **state,
            "output": selected_output,
            "final_db": selected_db,
            "final_sql": extracted_sql,
            }
        else:
            print(f"❌ No match for GPT-selected DB: {selected_db_raw}")
            return {**state, "output": f"No valid database selected: {selected_db_raw}"}
    except Exception as e:
        print(f"⚠️ Failed to select best answer: {e}")
        return {**state, "output": "Error selecting best answer."}


def select_databases_with_gpt(state: QueryState) -> QueryState:
    print("🧠 select_databases_with_gpt")
    question = state["question"]
    from description_utils import load_descriptions
    descriptions = load_descriptions()

    # --- Step 1: use embeddings to narrow down to top-k ---
    db_ids = list(descriptions.keys())
    question_embedding = model.encode(question, convert_to_tensor=True)

    scores = []
    for db_id in db_ids:
        desc_text = descriptions[db_id].get("description", "")
        desc_embedding = model.encode(desc_text, convert_to_tensor=True)
        score = util.pytorch_cos_sim(question_embedding, desc_embedding).item()
        scores.append((score, db_id))

    top_k = 8
    top_dbs = [db_id for _, db_id in sorted(scores, reverse=True)[:top_k]]

    # --- Step 2: format a short GPT prompt using only those ---
    top_descriptions = "\n".join(
        [f"- {db_id}: {descriptions[db_id]['description']}" for db_id in top_dbs]
    )

    prompt = f"""
You are helping choose the most relevant databases to answer this question:

\"\"\"{question}\"\"\"

Here are descriptions of candidate databases:
{top_descriptions}

Return a JSON list (e.g., ["db1", "db2"]) of the most relevant databases in order of relevance.
""".strip()

    try:
        gpt_response, _ = run_gpt35(prompt)
        selected = json.loads(gpt_response)
        print(f"✅ GPT selected DBs: {selected}")
        return {**state, "gpt_selected_dbs": selected}
    except Exception as e:
        print(f"⚠️ GPT DB selection failed: {e}")
        return {**state, "gpt_selected_dbs": top_dbs}  # fallback to top-k



def generate_sql_multi(state: QueryState) -> QueryState:
    print("🧠 generate_sql_multi")

    question = state["question"]
    dbs = state.get("dbs", [])
    all_outputs = {}

    for db_id in dbs:
        print(f"📊 Trying DB: {db_id}")
        try:
            sql, result = run_query(db_id, question)

            # Ensure both sql and result are strings
            result_str = str(result) if result is not None else "(No result returned)"
            sql_str = str(sql) if sql is not None else "SQL generation failed"

            # Safely combine
            combined = result_str + f"\n\n📝 SQL used:\n{sql_str}"
            all_outputs[db_id] = combined

        except Exception as e:
            all_outputs[db_id] = f"[Execution error on {db_id}] {e}"

    return {**state, "all_outputs": all_outputs}




def final_output(state: QueryState) -> QueryState:
    print("🏁 final_output")

    raw = state.get("output", "")
    question = state.get("question", "")
    formatted = format_with_gpt(raw, question)

    final_db = state.get("final_db", "Unknown")
    final_sql = state.get("final_sql", "SQL not found")
    gpt_candidates = state.get("gpt_selected_dbs", [])

    print("\n🟢 Final Answer:\n" + formatted)
    print("\n📦 Found in DB:", final_db)
    print("\n🧾 SQL Used:\n" + final_sql)
    print("\n🎯 GPT Candidate DBs:", gpt_candidates)

    return {
        **state,
        "output": formatted,
        "final_db": final_db,
        "final_sql": final_sql,
        "gpt_selected_dbs": gpt_candidates
    }







def build_graph():
    graph = StateGraph(QueryState)

    graph.add_node("extract_context", extract_context)
    graph.add_node("select_databases_with_gpt", select_databases_with_gpt)
    graph.add_node("retrieve_schema", retrieve_schema)
    graph.add_node("generate_sql_multi", generate_sql_multi)
    graph.add_node("select_best_answer", select_best_answer)
    graph.add_node("final_output", final_output)

    graph.set_entry_point("extract_context")
    graph.add_edge("extract_context", "select_databases_with_gpt")
    graph.add_edge("select_databases_with_gpt", "retrieve_schema")
    graph.add_edge("retrieve_schema", "generate_sql_multi")
    graph.add_edge("generate_sql_multi", "select_best_answer")
    graph.add_edge("select_best_answer", "final_output")

    graph.set_finish_point("final_output")
    return graph.compile()
