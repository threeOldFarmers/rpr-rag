from .base_llm_model import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from openai import OpenAI
import re

class QWQ32B(BaseLanguageModel):

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.maximun_token = 98304

    def tokenize(self, text):
        token_count = len(text) * 0.3
        return token_count

    def prepare_for_inference(self, **model_kwargs):
        self.client = OpenAI(
            api_key='YOUR_API_KEY',
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )


    def strip_braces(self, text):
        return re.sub(r"\{(.*?)\}", r"\1", text)

    def generate_sentence(self, llm_input, question=""):
        llm_input = self.strip_braces(llm_input)
        messages = [{"role": "user", "content": llm_input}]
        retry = True
        while retry:
            retry = False
            try:
                completion = self.client.chat.completions.create(
                    model="qwq-32b",
                    messages=messages,
                    stream=True,
                )

                answer_content = ""

                for chunk in completion:
                    if not chunk.choices:
                        print("\nUsage:")
                        print(chunk.usage)
                        continue

                    delta = chunk.choices[0].delta

                    if hasattr(delta, "content") and delta.content:
                        answer_content += delta.content

                return answer_content
            except Exception as e:
                error_message = str(e)
                if 'timeout' in error_message.lower():
                    retry = True
                else:
                    print(e)
                    return f"request error: {e}"


