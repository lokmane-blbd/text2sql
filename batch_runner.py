import os
from main import run_query  # assuming you refactor main.py to expose a function

def list_databases(base_path="spider/databases"):
    return sorted([
        name for name in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, name))
    ])

if __name__ == "__main__":
    user_question = "Show all students enrolled in 2020."

    all_dbs = list_databases()
    for db_id in all_dbs:
        print("="*60)
        print(f"🟡 Running on database: {db_id}")
        print("-"*60)
        try:
            run_query(db_id, user_question)
        except Exception as e:
            print(f"❌ Failed on {db_id}: {e}")
        print("="*60)
