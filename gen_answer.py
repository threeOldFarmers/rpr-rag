import os
import json
import argparse

from datasets import load_dataset
from tqdm import tqdm
from llm_models import get_registered_model
from utils import *
from prompt_builder import PromptBuilder
from evaluate_answer import eval_result

def prediction(data, input_builder, model):
    qid = data["id"]
    question = data["question"]
    truth = data["a_entity"]
    input = input_builder.process_input(data)
    prediction = model.generate_sentence(input, question)
    if prediction is None:
        return None
    result = {
        "id": qid,
        "question": question,
        "prediction": prediction,
        "truth": truth,
        "input": input,
    }
    return result

def generate_answer(args, LLM):
    if args.with_rag:
        output_dir = os.path.join(args.output_path, args.d, args.model_name+"+rag", args.split)
    else:
        output_dir = os.path.join(args.output_path, args.d, args.model_name, args.split)
    print("Save results to: ", output_dir)

    # Load dataset
    print("Loading dataset: ", args.data_path)
    dataset = load_dataset("json", data_files=args.data_path, split="train")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = LLM(args)
    input_builder = PromptBuilder(
        prompt_path=args.prompt_path,
        with_rag=args.with_rag,
        cot=args.cot,
        explain=args.explain,
        each_line=args.each_line,
        maximun_token=model.maximun_token,
        tokenize=model.tokenize,
    )
    model.prepare_for_inference()

    prediction_file = os.path.join(output_dir, f"answer.jsonl")
    f, processed_results = get_output_file(prediction_file, force=args.force)
    for data in tqdm(dataset):
        qid = data["id"]

        if qid in processed_results:
            continue

        res = prediction(data, input_builder, model)

        if args.debug:
            print("res: ", res)
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
        f.flush()
    f.close()

    eval_result(prediction_file)

    return prediction_file

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="data/metaqa-1hop_test_dataset.jsonl"
    )
    parser.add_argument("--d", "-d", type=str, default="metaqa-1hop")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
    )
    parser.add_argument("--output_path", type=str, default="results/answer")
    parser.add_argument(
        "--model_name",
        type=str,
        help="model_name for save results",
        default="none",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="HUGGING FACE MODEL or model path",
        default="none",
    )
    parser.add_argument(
        "--prompt_path", type=str, help="prompt_path", default="prompts/gen_answer.txt"
    )
    parser.add_argument("--with_rag", action="store_true", help="Use RAG", default=False)
    parser.add_argument("--explain", action="store_true", help="With Explain", default=False)
    parser.add_argument("--each_line", action="store_true", help="Answer In Each Line", default=False)
    parser.add_argument(
        "--force", "-f", action="store_true", help="force to overwrite the results"
    )
    parser.add_argument("--debug", action="store_true", help="Debug")
    parser.add_argument("--cot", action="store_true", help="CoT", default=False)

    args = parser.parse_args()
    LLM = get_registered_model(args.model_name)
    LLM.add_args(parser)
    args = parser.parse_args()

    gen_path = generate_answer(args, LLM)
