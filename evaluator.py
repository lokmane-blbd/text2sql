import sqlparse

def evaluate_sql_outputs(question, gpt_output, gold=None, verbose=False):
    if not verbose:
        return

    print("\n" + "="*50)
    print(f"Question: {question}")
    print("-"*50)
    print("GPT-3.5 Output SQL:")
    print(sqlparse.format(gpt_output, reindent=True))
    if gold:
        print("-"*50)
        print("Gold SQL:")
        print(sqlparse.format(gold, reindent=True))
    print("="*50 + "\n")
