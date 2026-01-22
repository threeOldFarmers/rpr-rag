from .base_llm_model import BaseLanguageModel
from .deepseek_reasoner import DeepseekReasoner
from .deepseek_v3 import DeepseekV3
from .deepseek_r1_7b import DeepseekR17B
from .deepseek_r1_8b import DeepseekR18B
from .llama4_17b import Llama417B
from .qwq_32b import QWQ32B
from .gpt4 import GPT4
from .gpt5 import GPT5
from .gpt5mini import GPT5mini

registered_llm_models = {
    'deepseek-reasoner': DeepseekReasoner,  #by deepseek api
    'deepseek-v3': DeepseekV3,              #by deepseek api
    'deepseek-r1-7b': DeepseekR17B,         #by aliyun api
    'deepseek-r1-8b': DeepseekR18B,         #by aliyun api
    'qwq-32b': QWQ32B,                      #by aliyun api
    'llama4-17b': Llama417B,                #by aliyun api
    'gpt-4': GPT4,                          #by openai api
    'gpt-5': GPT5,                          #by openai api
    'gpt-5-mini': GPT5mini,                 #by openai api
}

def get_registered_model(model_name) -> BaseLanguageModel:
    for key, value in registered_llm_models.items():
        if key == model_name:
            return value
    raise ValueError(f"No registered model found for name {model_name}")
