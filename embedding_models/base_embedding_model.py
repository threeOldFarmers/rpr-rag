
class BaseEmbeddingModel(object):

    def __init__(self):
        self.load_model()

    def load_model(self):
        raise NotImplementedError

    def get_dimension(self):
        raise NotImplementedError

    def embed_question(self, question):
        raise NotImplementedError

    def embed_nodes(self, nodes):
        raise NotImplementedError

    def embed_relations(self, relations):
        raise NotImplementedError

