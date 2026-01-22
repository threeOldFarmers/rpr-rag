import os
import json
import argparse

from datasets import load_dataset
from tqdm import tqdm
from decomposer import *
from llm import get_llm
from utils import *

def generate_reasoning_chain(args):
    llm = get_llm()

    # input_file = os.path.join(args.data_path, args.d)
    output_dir = os.path.join(args.output_path, args.d, args.model_name, args.split)
    print("Save results to: ", output_dir)

    # Load dataset
    print("Loading dataset: ", args.data_path)
    dataset = load_dataset("json", data_files=args.data_path, split="train")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prediction_file = os.path.join(output_dir, f"reasoning_chain.jsonl")
    f, processed_results = get_output_file(prediction_file, force=args.force)
    for data in tqdm(dataset):
        question = data["question"]
        # input_text = data["text"]
        qid = data["id"]
        if qid in processed_results:
            continue
        sub_questions = question_decompose(llm, question)
        if args.debug:
            print("ID: ", qid)
            print("Question: ", question)
            print("Sub Questions: ", sub_questions)
        data = {
            "id": qid,
            "question": question,
            "sub_questions": sub_questions,
            # "ground_paths": data["ground_paths"],
        }
        f.write(json.dumps(data) + "\n")
        f.flush()
    f.close()

    return prediction_file

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="data/webqsp_test_dataset.jsonl"
    )
    parser.add_argument("--d", "-d", type=str, default="webqsp")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
    )
    parser.add_argument("--output_path", type=str, default="results/reasoning_chain")
    parser.add_argument(
        "--model_name",
        type=str,
        help="model_name for save results",
        default="deepseek-reasoner",
    )
    parser.add_argument(
        "--force", "-f", action="store_true", help="force to overwrite the results"
    )
    parser.add_argument("--debug", action="store_true", help="Debug")

    args = parser.parse_args()

    gen_path = generate_reasoning_chain(args)
