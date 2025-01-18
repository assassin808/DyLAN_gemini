import ast
import json
import os
# import openai
import random
import sys
import threading
import concurrent.futures
from prettytable import PrettyTable
from LLMLP import LLMLP  # Assuming LLMLP is your custom class
from utils import *

# Configure OpenAI API (replace with your actual credentials)
# openai.api_key = "YOUR_API_KEY"
# openai.api_base = "YOUR_API_BASE"  # If using a different base URL
# openai.api_type = "azure"  # Or "openai" if not using Azure
# openai.api_version = "YOUR_API_VERSION"

# Default values (can be overridden per question)
EXP_NAME = "trial_1"
MODEL = "chatgpt0301"
ACTIVATION = "listwise"
TYPE = "open-ended"
DIR_NAME = "trial"

# Roles for the LLM agents
if sys.argv[2] == "multi": 
    ROLES = ["Historian", "Mathematician", "Psychologist", "Programmer"]
else:
    ROLES = ["Assistant", "Assistant", "Assistant", "Assistant"]

def set_rd_seed(seed):
    random.seed(seed)

def process_question(question_data, output_filename="output2.json"):
    """
    Processes a single question using LLMLP and saves the results.
    """
    set_rd_seed(0)  # You might want to use a different seed per thread or question

    question = question_data['question']
    skill = question_data.get('skill', [])
    print(f"\nProcessing question: {question}\n")

    # Get parameters from question_data, or use defaults
    roles = question_data.get('roles', ROLES)
    model = question_data.get('model', MODEL)
    activation = question_data.get('activation', ACTIVATION)
    type_ = question_data.get('type', TYPE)
    num_rounds = question_data.get('num_rounds',3)

    llmlp = LLMLP(model, len(roles), roles, num_rounds, activation, type_, model)

    llmlp.zero_grad()
    res, resp_cnt, completions, prompt_tokens, completion_tokens = llmlp.forward(question)
    imp_score = llmlp.backward(res)
    imp_score = [[imp_score[idx] for idx in range(len(roles) * rid, len(roles) * (rid + 1))] for rid in range(num_rounds)]

    pt = PrettyTable()
    pt.add_column("Round", roles)
    for rid in range(num_rounds):
        responses = [(completions[idx][rid] if completions[idx][rid] is not None else "No response.") for idx in range(len(roles))]
        pt.add_column(str(rid + 1), responses, "l")

    print(f"#API calls: {resp_cnt}")
    print(f"Final Answer: {res}")

    ans = {
        "question": question,
        "skill": skill,
        "summarized_answer": res,
        "round":1,
        "prompt_tokens": prompt_tokens,
        "api_calls": resp_cnt,
        "completion_tokens": completion_tokens,
        "rounds": {f"round_{rid+1}": [completions[idx][rid] for idx in range(len(roles))] for rid in range(num_rounds)}
    }

    with threading.Lock():  # Use a lock to protect file access
        with open(output_filename, 'a') as f:
            json.dump(ans, f)
            f.write('\n')

def read_json_file(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

def main(file_path):
    data = read_json_file(file_path)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit each question to the thread pool
        futures = [executor.submit(process_question, question_data) for question_data in data]

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)

    print("All questions processed.")

if __name__ == "__main__":
    main(sys.argv[1])