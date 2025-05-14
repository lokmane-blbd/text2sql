from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langgraph_workflow import build_graph

app = FastAPI()
graph = build_graph()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    db_id: str = None  # Optional, override if needed

@app.post("/query")
def query_handler(data: QueryRequest):
    try:
        initial_state = {
            "question": data.question,
            "dbs": [data.db_id] if data.db_id else [],
            "output": "",
            "attempt": 0,
            "hinted_db": None,
            "hinted_column": None,
            "gpt_selected_dbs": [],
            "all_outputs": {},
            "final_db": None,
            "final_sql": None
        }

        result = graph.invoke(initial_state)

        return {
        "result": result.get("output", "No meaningful answer found."),
        "db": result.get("final_db", None),
        "sql": result.get("final_sql", None),
        "candidates": result.get("gpt_selected_dbs", [])
        }

    except Exception as e:
        return {"error": str(e)}
