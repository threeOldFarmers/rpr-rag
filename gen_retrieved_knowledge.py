import os
import json
import argparse

from datasets import load_dataset
from tqdm import tqdm
from retriever import *
from utils import *
from llm_models import get_registered_model
from embedding_models import get_registered_Embedding_model
from prompt_builder import PromptBuilder
from question_utils import extract_key_phrases

def merge_reasoning_chain_result(qa_dataset, chain_dataset, n_proc=1, filter_empty=False, least_length=1):
    question_to_sub = {}
    for data in chain_dataset:
        qid = data["id"]
        original_question = data["question"]
        sub_questions = data["sub_questions"]
        # Pad sub_questions if needed
        if len(sub_questions) < least_length:
            sub_questions += [original_question] * (least_length - len(sub_questions))

        question_to_sub[qid] = {
            "sub_questions": sub_questions,
        }

    def find_sub(sample):
        qid = sample["id"]
        sample["sub_questions"] = []
        if qid in question_to_sub:
            sample["sub_questions"] = question_to_sub[qid]["sub_questions"]
        return sample

    qa_dataset = qa_dataset.map(find_sub, num_proc=n_proc)
    if filter_empty:
        qa_dataset = qa_dataset.filter(
            lambda x: len(x["sub_questions"]) > 0, num_proc=n_proc
        )
    return qa_dataset


def generate_retrieved_knowledge(args, LLM, EM):
    # input_file = os.path.join(args.data_path, args.d)
    output_dir = os.path.join(args.output_path, args.d, args.model_name, args.split)
    print("Save results to: ", output_dir)

    # Load dataset
    print("Loading dataset: ", args.data_path)
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    if args.with_reasoning_chain:
        rule_dataset = load_json(args.reasoning_path)
        dataset = merge_reasoning_chain_result(dataset, rule_dataset, 1, least_length=2)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    embeddingModel = EM()
    model = LLM(args)
    input_builder = PromptBuilder(
        prompt_path=args.prompt_path,
        maximun_token=model.maximun_token,
        tokenize=model.tokenize,
    )
    model.prepare_for_inference()

    def extract_score(text):
        match = re.search(r"(\d+)/10", text)

        return int(match.group(1)) if match else 0

    def evaluate_with_llm(question, retrieved_knowledge):
        input = input_builder.process_re(question, retrieved_knowledge)
        try:
            prediction = model.generate_sentence(input, question)
        except Exception as e:
            log_error(e)
            prediction = f"""
            <Question1>{e}</Question1>
            <Question1>5/10</Question1>
            <Question1>No</Question1>
            """

        response_text = prediction.strip()
        response_text = "\n".join(line for line in response_text.splitlines() if line.strip())
        log_info(f"response_text: {response_text}")
        if 'inappropriate content' in response_text.lower():
            answer1 = "Good"
            answer2 = "7/10"
            answer3 = "Yes"
        else:
            answer1 = ""
            answer2 = "0/10"
            answer3 = "No"
        match = re.search(r"<Question1>(.*?)</Question1>", response_text)
        if match:
            answer1 = match.group(1)
        match = re.search(r"<Question2>(.*?)</Question2>", response_text)
        if match:
            answer2 = match.group(1)
        match = re.search(r"<Question3>(.*?)</Question3>", response_text)
        if match:
            answer3 = match.group(1)
        log_info(f"answer1: {answer1}")
        log_info(f"answer2: {answer2}")
        log_info(f"answer3: {answer3}")

        confidence_score = extract_score(answer2.strip())
        sufficiency = "yes" in answer3.lower()

        return confidence_score, sufficiency

    prediction_file = os.path.join(output_dir, f"retrieved_knowledge.jsonl")
    f, processed_results = get_output_file(prediction_file, force=args.force)
    for data in tqdm(dataset):
        qid = data["id"]
        relations = data["graph"]
        q_entity = data["q_entity"]
        a_entity = data["a_entity"]
        original_question = data["question"]
        if args.with_reasoning_chain:
            sub_questions = data["sub_questions"]
        else:
            sub_questions = [original_question, original_question, original_question, original_question]
        if args.with_question_extracted:
            sub_questions = [extract_key_phrases(q) for q in sub_questions]

        if qid in processed_results:
            continue

        reconstruct_graph_embedding(relations, embeddingModel)
        retrieved_knowledge, count = do_retrieve(original_question, q_entity, sub_questions, a_entity, evaluate_with_llm, embeddingModel)

        if args.debug:
            print("ID: ", qid)
            print("Question: ", original_question)
            print("Sub Questions: ", sub_questions)
            print("Retrieved Knowledge: ", retrieved_knowledge)
            print("Retrieve Count: ", count)
        data = {
            "id": qid,
            "question": original_question,
            "q_entity": q_entity,
            "a_entity": a_entity,
            "sub_questions": sub_questions,
            "retrieved_knowledge": retrieved_knowledge,
            "retrieve_count": count
        }
        f.write(json.dumps(data) + "\n")
        f.flush()
    f.close()

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
    parser.add_argument("--output_path", type=str, default="results/retrieved_knowledge")
    parser.add_argument("--with_reasoning_chain", action="store_true", help="Use Reasoning Chain", default=False)
    parser.add_argument("--with_question_extracted", action="store_true", help="Extract Question Key Phrases", default=False)
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        help="embedding model name",
        default="none",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="llm model name",
        default="none",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="HUGGING FACE MODEL or model path",
        default="none",
    )
    parser.add_argument(
        "--prompt_path", type=str, help="prompt_path", default="prompts/eval_retrieve.txt"
    )
    parser.add_argument(
        "--force", "-f", action="store_true", help="force to overwrite the results"
    )
    parser.add_argument("--debug", action="store_true", help="Debug")

    args = parser.parse_args()
    EM = get_registered_Embedding_model(args.embedding_model_name)
    LLM = get_registered_model(args.model_name)
    LLM.add_args(parser)
    args = parser.parse_args()

    gen_path = generate_retrieved_knowledge(args, LLM, EM)
