from .base_embedding_model import BaseEmbeddingModel
from .bge_small_en import BgeSmallEn
from .webqsp_bge_small_en import WebQSPBgeSmallEn
from .cwq_bge_small_en import CWQBgeSmallEn

registered_embedding_models = {
    'bge-small-en': BgeSmallEn,
    'webqsp-bge-small-en': WebQSPBgeSmallEn,
    'cwq-bge-small-en': CWQBgeSmallEn,
}

def get_registered_Embedding_model(model_name) -> BaseEmbeddingModel:
    for key, value in registered_embedding_models.items():
        if key == model_name:
            return value
    raise ValueError(f"No registered model found for name {model_name}")