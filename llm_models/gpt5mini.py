from .base_llm_model import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
import re

class GPT5mini(BaseLanguageModel):

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.maximun_token = 64000

    def tokenize(self, text):
        token_count = len(text)*0.3
        return token_count

    def prepare_for_inference(self, **model_kwargs):
        self.llm = BaseChatOpenAI(
            model='gpt-5-mini',
            base_url='https://api.openai.com/v1',
            api_key='YOUR_API_KEY',
            # max_tokens=1024,
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
                if 'RequestTimeOut' in error_message or 'Internal Server Error' in error_message or 'Connection Error' in error_message:
                    retry = True
                else:
                    print(e)
                    return f"request error: {e}"
