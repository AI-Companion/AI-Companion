import json


class SentimentAnalysisConfigReader():
    """
    Loads sentiment analysis use case parameters from a .json file
    """
    def __init__(self, file_name):
        self.config = self.read_file(file_name)

    def read_file(self, file_name):
        """
        used to load configuration file
        Args:
            file_name: relative path for the configuration .json file
        """
        with open(file_name, "r") as f:
            return json.load(f)

    @property
    def dataset_name(self):
        """
        Name of the dataset to be used
        """
        return self.config["dataset_name"]

    @property
    def dataset_url(self):
        """
        Web url for the training data
        """
        return self.config["dataset_url"]

    @property
    def embeddings_path_glove(self):
        """
        Web url for embedding matrix. will be used in case a pretraining embedding is required
        """
        return self.config["embeddings_path_glove"]

    @property
    def embeddings_path_fasttext(self):
        """
        Web url for embedding matrix. will be used in case a pretraining embedding is required
        """
        return self.config["embeddings_path_fasttext"]

    @property
    def pre_trained_embedding(self):
        """
        Wether to use a pretrained embedding or not.
        in case yes, a pretrained embedding will be downloaded from the internet
        """
        return self.config["pre_trained_embedding"]

    @property
    def embedding_algorithm(self):
        """
        Not in use currently, relates to the embedding algorithm to use
        """
        return self.config["embedding_algorithm"]

    @property
    def model_optimizer(self):
        """
        not in use currently, related to gradient descent optimizer to be used
        """
        return self.config["model_optimizer"]

    @property
    def vocab_size(self):
        """
        total number of unique tokens to perform the training
        """
        return self.config["vocab_size"]

    @property
    def validation_split(self):
        """
        ratio of split into validation and training
        """
        return self.config["validation_split"]

    @property
    def max_sequence_length(self):
        """
        maximum tolerated number of characters in a sentence. any length higher will be clipped
        any length shorter will be padded
        """
        return self.config["max_sequence_length"]

    @property
    def embedding_dimension(self):
        """
        sets the size of the embeddings. possible embeddings are [50, 100, 200, 300]
        """
        return self.config["embedding_dimension"]

    @property
    def experimental_mode(self):
        """
        enable if a quick training is required. will select random 2000 observation
        """
        return self.config["experimental_mode"]

    @property
    def model_name(self):
        """
        enables a selection between rnn model of tf-idf followed by classical ml algorithm
        """
        return self.config["model_name"]

    @property
    def eval_model_name(self):
        """
        selects the name of the model to perform evaluation on
        """
        return self.config["eval_model_name"]


class NamedEntityRecognitionConfigReader():
    """
    Loads named entity recognition use case parameters from a .json file
    """
    def __init__(self, file_name):
        self.config = self.read_file(file_name)

    def read_file(self, file_name):
        """
        used to load configuration file
        Args:
            file_name: relative path for the configuration .json file
        """
        with open(file_name, "r") as f:
            return json.load(f)

    @property
    def dataset_name(self):
        """
        name of the dataset to be used
        """
        return self.config["dataset_name"]

    @property
    def dataset_url(self):
        """
        web url for the training data
        """
        return self.config["dataset_url"]

    @property
    def validation_split(self):
        """
        ratio of split into validation and training
        """
        return self.config["validation_split"]

    @property
    def max_sequence_length(self):
        """
        maximum tolerated number of characters in a sentence. any length higher will be clipped
        any length shorter will be padded
        """
        return self.config["max_sequence_length"]

    @property
    def model_name(self):
        """
        sets the size of the embeddings. possible embeddings are [50, 100, 200, 300]
        """
        return self.config["model_name"]

    @property
    def experimental_mode(self):
        """
        enable if a quick training is required. will select random 2000 observation
        """
        return self.config["experimental_mode"]

    @property
    def vocab_size(self):
        """
        total number of unique tokens to perform the training
        """
        return self.config["vocab_size"]

    @property
    def pre_trained_embedding(self):
        """
        weather to use a pretrained embedding or not.
        in case yes, a pretrained embedding will be downloaded from the internet
        """
        return self.config["pre_trained_embedding"]

    @property
    def embedding_dimension(self):
        """
        sets the size of the embeddings. possible embeddings are [50, 100, 200, 300]
        """
        return self.config["embedding_dimension"]

    @property
    def embedding_algorithm(self):
        """
        not in use currently, relates to the embedding algorithm to use
        """
        return self.config["embedding_algorithm"]

    @property
    def embeddings_path_glove(self):
        """
        web url for embedding matrix. will be used in case a pretraining embedding is required
        """
        return self.config["embeddings_path_glove"]

    @property
    def embeddings_path_fasttext(self):
        """
        web url for embedding matrix. will be used in case a pretraining embedding is required
        """
        return self.config["embeddings_path_fasttext"]
