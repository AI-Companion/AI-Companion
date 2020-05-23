import json

class ConfigReader():
    def __init__(self, filename):
        self.config = self.read_file(filename)

    def read_file(self, filename):
        with open(filename, "r") as f:
            return json.load(f)

    @property
    def dataset_name(self):
        return self.config["dataset_name"]

    @property
    def dataset_url(self):
        return self.config["dataset_url"]

    @property
    def embeddings_path(self):
        return self.config["embeddings_path"]

    @property
    def pre_trained_embedding(self):
        return self.config["pre_trained_embedding"]

    @property
    def embedding_algorithm(self):
        return self.config["embedding_algorithm"]

    @property
    def model_optimizer(self):
        return self.config["model_optimizer"]

    @property
    def vocab_size(self):
        return self.config["vocab_size"]

    @property
    def validation_split(self):
        return self.config["validation_split"]

    @property
    def max_sequence_length(self):
        return self.config["max_sequence_length"]

    @property
    def embedding_dimension(self):
        return self.config["embedding_dimension"]

    @property
    def experimental_mode(self):
        return self.config["experimental_mode"]

