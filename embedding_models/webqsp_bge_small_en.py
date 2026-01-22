from .base_embedding_model import BaseEmbeddingModel
from langchain_huggingface import HuggingFaceEmbeddings

class WebQSPBgeSmallEn(BaseEmbeddingModel):

    def load_model(self):
        self.embeddingModel = HuggingFaceEmbeddings(model_name="embedding_models/webqsp-embedding-model")

    def get_dimension(self):
        return 384

    def embed_question(self, question):
        return self.embeddingModel.embed_query(question)

    def embed_nodes(self, nodes):
        return self.embeddingModel.embed_documents(nodes)

    def embed_relations(self, relations):
        return self.embeddingModel.embed_documents(relations)


