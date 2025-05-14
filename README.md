## 📁 Project Structure

| File/Folder | Purpose |
|-------------|---------|
| `api.py` | FastAPI server |
| `main.py` | Terminal-based CLI |
| `langgraph_workflow.py` | LangGraph agent pipeline |
| `model_runner.py` | Runs GPT-3.5 for prompts |
| `schema_utils.py` | Loads and parses schema |
| `description_utils.py` | Loads database descriptions |
| `vector_store.py` | RAG retriever (chunk embedding + similarity) |
| `run_all.sh` | Starts server + ngrok + CLI input |
| `index.html` | Minimal web frontend |
| `requirements.txt` | Python dependencies |
| `spider/` | SQLite databases from Spider dataset |
| `schema_embeddings/` | Precomputed DB embeddings |
| `descriptions.json` | GPT-friendly descriptions per DB |
| `results_log.csv` | Logs answers and SQL for review |

---

## Setup Instructions

### 1. 🔁 Clone This Repo
### 2. Create Environment
conda create -n text2sql python=3.10

conda activate text2sql


3.  Install Requirements


pip install -r requirements.txt

export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxx



4-----




One-Time: Precompute Schema Embeddings

Before using the system, embed all schemas:

run: python precompute_schema_embeddings.py
This will fill the schema_embeddings/ folder.



5-------




 CLI + Ngrok Combo 
chmod +x run_all.sh
./run_all.sh

