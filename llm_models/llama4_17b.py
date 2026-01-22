from http import HTTPStatus

from .base_llm_model import BaseLanguageModel
import dashscope
import time

class Llama417B(BaseLanguageModel):

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.maximun_token = 8000

    def tokenize(self, text):
        token_count = len(text)*0.3
        return token_count

    def prepare_for_inference(self, **model_kwargs):
        pass

    def generate_sentence(self, llm_input, question=""):
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": llm_input}
                ]
            }
        ]
        retry = True
        while retry:
            retry = False
            #API request: no more than 10 per second
            time.sleep(6)
            response = dashscope.MultiModalConversation.call(
                api_key="YOUR_API_KEY",
                model='llama-4-scout-17b-16e-instruct',
                messages=messages,
                # result_format='message',  # set the result to be "message" format.
            )

            if response.status_code == HTTPStatus.OK:
                return response.output.choices[0].message.content[0]["text"]
            elif response.status_code == HTTPStatus.REQUEST_TIMEOUT:
                retry = True
            elif 'quota exceed' in response.message:
                retry = True
            else:
                print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                    response.request_id, response.status_code,
                    response.code, response.message
                ))
                return f"request error: {response.message}"
