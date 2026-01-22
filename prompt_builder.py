import random
from typing import Callable

class PromptBuilder(object):
    SAQ_INSTRUCTION = """Please answer the following questions. Please keep the answer as simple as possible and return all the possible answer as a list."""
    SAQ_RULE_INSTRUCTION = """
        Given the reasoning paths, identify and extract all relevant named entities (e.g., people, teams, organizations, locations, etc.) that help answer the question.
        Return only the entity names as a list. Be comprehensive — include all possible entities mentioned in the reasoning paths that could be considered answers to the question.
        Do not generalize (e.g., "sports" or "cities") — instead, return the specific entity names mentioned in the reasoning paths.
        Do not explain your answer.
    """
    COT = """ Let's think it step by step."""
    EXPLAIN = """ Please explain your answer."""
    QUESTION = """Question:\n{question}"""
    GRAPH_CONTEXT = """Reasoning Paths:\n{context}\n\n"""
    CHOICES = """\nChoices:\n{choices}"""
    EACH_LINE = """ Please return each answer in a new line."""

    def __init__(self, prompt_path, with_rag=False, cot=False, explain=False,
                 each_line=False, maximun_token=4096, tokenize: Callable = lambda x: len(x)):
        self.prompt_template = self._read_prompt_template(prompt_path)
        self.with_rag = with_rag
        self.cot = cot
        self.explain = explain
        self.maximun_token = maximun_token
        self.tokenize = tokenize
        self.each_line = each_line

    def _read_prompt_template(self, template_file):
        with open(template_file) as fin:
            prompt_template = f"""{fin.read()}"""
        return prompt_template

    def process_re(self, question, retrieved_knowledge):
        other_prompt = self.prompt_template.format(question=question, retrieved_knowledge="")
        clip_re = self.check_prompt_length(other_prompt, retrieved_knowledge, self.maximun_token)
        input = self.prompt_template.format(question=question, retrieved_knowledge=clip_re)

        return input

    def process_input(self, question_dict):
        question = question_dict['question']

        if not question.endswith('?'):
            question += '?'

        input = self.QUESTION.format(question=question)
        if self.with_rag:
            instruction = self.SAQ_RULE_INSTRUCTION
        else:
            instruction = self.SAQ_INSTRUCTION

        if self.cot:
            instruction += self.COT

        if self.explain:
            instruction += self.EXPLAIN

        if self.each_line:
            instruction += self.EACH_LINE

        if self.with_rag:
            other_prompt = self.prompt_template.format(instruction=instruction,
                                                       input=self.GRAPH_CONTEXT.format(context="") + input)
            context = self.check_prompt_length(other_prompt, question_dict["retrieved_knowledge"], self.maximun_token)

            input = self.GRAPH_CONTEXT.format(context=context) + input

        input = self.prompt_template.format(instruction=instruction, input=input)

        return input

    def check_prompt_length(self, prompt, retrieved_knowledge, maximun_token):
        '''Check whether the input prompt is too long. If it is too long, remove the first path and check again.'''
        # all_paths = "\n".join(list_of_paths)
        all_tokens = prompt + retrieved_knowledge
        if self.tokenize(all_tokens) < maximun_token:
            return retrieved_knowledge
        else:
            # Shuffle the paths
            reasoning_paths = retrieved_knowledge.split("\n")
            random.shuffle(reasoning_paths)
            new_reasoning_paths = []
            # check the length of the prompt
            for p in reasoning_paths:
                tmp_all_paths = "\n".join(new_reasoning_paths + [p])
                tmp_all_tokens = prompt + tmp_all_paths
                if self.tokenize(tmp_all_tokens) > maximun_token:
                    return "\n".join(new_reasoning_paths)
                new_reasoning_paths.append(p)
