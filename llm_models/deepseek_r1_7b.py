from .base_llm_model import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
import re

class DeepseekR17B(BaseLanguageModel):

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.maximun_token = 32768

    def tokenize(self, text):
        token_count = len(text) * 0.5
        return token_count

    def prepare_for_inference(self, **model_kwargs):
        self.llm = BaseChatOpenAI(
            model='deepseek-r1-distill-qwen-7b',
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
            api_key='YOUR_API_KEY',
        )

    def strip_braces(self, text):
        return re.sub(r"\{(.*?)\}", r"\1", text)

    def generate_sentence(self, llm_input, question=""):
        llm_input = self.strip_braces(llm_input)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", llm_input),
                ("human", "{question}"),
            ]
        )

        chain = prompt | self.llm
        retry = True
        while retry:
            retry = False
            try:
                answer = chain.invoke(
                    {
                        "input_language": "English",
                        "output_language": "English",
                        "question": question,
                    }
                )
                return answer.content
            except Exception as e:
                error_message = str(e)
                if 'RequestTimeOut' in error_message:
                    retry = True
                else:
                    print(e)
                    return f"request error: {e}"
