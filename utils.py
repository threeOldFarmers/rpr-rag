import os
import json
import re
import huggingface_hub

def set_proxy():
    http_proxy = 'http://127.0.0.1:7890'
    os.environ['HTTP_PROXY'] = http_proxy
    os.environ['HTTPS_PROXY'] = http_proxy

def set_huggingface():
    huggingface_hub.login("hugging face key")

def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, []
    else:
        with open(path, "r") as f:
            processed_results = []
            for line in f:
                try:
                    results = json.loads(line)
                except:
                    raise ValueError("Error in line: ", line)
                processed_results.append(results["id"])
        fout = open(path, "a")
        return fout, processed_results

def load_json(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def trim_question(question, entity):
    entity_lower = entity.lower()
    question_lower = question.lower()

    trimmed_question = re.sub(r'\b' + re.escape(entity_lower) + r'\b', '', question_lower,
                              flags=re.IGNORECASE).strip()

    trimmed_question = re.sub(r'\s+', ' ', trimmed_question)

    return trimmed_question

