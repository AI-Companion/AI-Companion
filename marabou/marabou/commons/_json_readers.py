"""
Contains abstract dataset class and abstract json configuration
"""
from abc import ABC, abstractmethod
import os
import subprocess
import pandas as pd
import json


class BaseConfigReader():
    """
    Parent class to load use case parameters from a .json file
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
    def model_optimizer(self):
        """
        not in use currently, related to gradient descent optimizer to be used
        """
        return self.config["model_optimizer"]

    @property
    def validation_split(self):
        """
        ratio of split into validation and training
        """
        return self.config["validation_split"]

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

    @property
    def h5_model_url(self):
        """
        gdrive fileid for the trained model
        """
        return self.config["h5_model_url"]

    @property
    def class_file_url(self):
        """
        gdrive fileid for the class file
        """
        return self.config["class_file_url"]

    @property
    def preprocessor_file_url(self):
        """
        gdrive fileid for the class file
        """
        return self.config["preprocessor_file_url"]

    @property
    def n_iter(self):
        """
        number of backprop iterations
        """
        return self.config["n_iter"]


class RNNConfigReader(BaseConfigReader):
    """
    Parent class for RNN use cases
    """

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
    def vocab_size(self):
        """
        total number of unique tokens to perform the training
        """
        return self.config["vocab_size"]

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
    def embeddings_path(self):
        """
        sets the path to the selected embedding algorithm
        """
        embeddings_path = self.embeddings_path_fasttext
        if self.embedding_algorithm == "glove":
            embeddings_path = self.embeddings_path_glove
        return embeddings_path


class NERConfigReader(RNNConfigReader):
    """
    Named Entity Recognition analysis json configuration file reader
    """
    pass


class SAConfigReader(RNNConfigReader):
    """
    Sentiment analysis json configuration file reader
    """
    pass


class TDConfigReader(RNNConfigReader):
    """
    Sentiment analysis json configuration file reader
    """
    pass
