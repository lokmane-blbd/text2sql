import csv
import os

def init_csv_log(filename="results_log.csv"):
    if not os.path.exists(filename):
        with open(filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["db_id", "question", "gpt_sql", "result", "error", "tokens"])


def log_result(db_id, question, sql, result, error=None, tokens=None, filename="results_log.csv"):
    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([db_id, question, sql, result, error, tokens])

