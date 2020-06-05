import subprocess
import os
import io
import numpy as np
from tensorflow.keras.layers import Embedding
from tensorflow.keras.initializers import Constant


class Glove6BEmbedding():
    """
    loads glove embedding from stanford
    """
    def __init__(self, embedding_dimension, word_index, vocab_size, embeddings_path, max_length):
        self.embedding_dimension = embedding_dimension
        self.word_index = word_index
        self.vocab_size = vocab_size
        self.embeddings_path = embeddings_path
        self.max_length = max_length
        self.embedding_layer = self.build_embedding()

    def get_embedding_matrix(self):
        """
        gets the embedding matrix according to the specified embedding dimension
        :return: None
        """
        print("===========> collecting pretrained embedding")
        script_path = os.path.join(os.getcwd(), "bash_scripts/load_stanford_6B_embedding.sh")
        subprocess.call("%s %s" % (script_path, self.embeddings_path), shell=True)
        file_name = 'glove.6B.100d.txt'
        if self.embedding_dimension in [50, 100, 200, 300]:
            file_name = 'glove.6B.%id.txt' % self.embedding_dimension
        file_url = os.path.join(os.getcwd(), 'embeddings', file_name)
        print("----> embedding file saved to %s" % file_url)
        embedding_index = {}
        with open(file_url) as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, 'f', sep=' ')
                embedding_index[word] = coefs
        embedding_matrix = np.zeros((self.vocab_size, self.embedding_dimension))
        for word, i in self.word_index.items():
            if i >= self.vocab_size:
                continue
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def build_embedding(self):
        """
        build the embedding layer which will be called by the model
        :return: An instance of tensorflow.keras.layers.Embedding
        """
        embeddings_matrix = self.get_embedding_matrix()
        embedding_layer = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dimension,
                                    embeddings_initializer=Constant(embeddings_matrix),
                                    input_length=self.max_length, trainable=False)
        return embedding_layer


class FastTextEmbedding():
    """
    loads fasttext embedding from facebook research
    @inproceedings{mikolov2018advances,
    title={Advances in Pre-Training Distributed Word Representations},
    author={Mikolov, Tomas and Grave, Edouard and Bojanowski, Piotr and Puhrsch, Christian and Joulin, Armand},
    booktitle={Proceedings of the International Conference on Language Resources and Evaluation (LREC 2018)},
    year={2018}
    }
    """
    def __init__(self, word_index, vocab_size, embeddings_path, max_length):
        self.word_index = word_index
        self.vocab_size = vocab_size
        self.embeddings_path = embeddings_path
        self.max_length = max_length
        self.embedding_dimension = 300
        self.embedding_layer = self.build_embedding()

    def get_embedding_matrix(self):
        """
        gets the embedding matrix according to the specified embedding dimension
        :return: None
        """
        print("===========> collecting pretrained embedding")
        script_path = os.path.join(os.getcwd(), "bash_scripts/load_fasttext_16B_embedding.sh")
        subprocess.call("%s %s" % (script_path, self.embeddings_path), shell=True)
        file_name = 'wiki-news-300d-1M.vec'
        file_url = os.path.join(os.getcwd(), 'embeddings', file_name)
        print("----> embedding file saved to %s" % file_url)
        fin = io.open(file_url, 'r', encoding='utf-8', newline='\n', errors='ignore')
        embedding_index = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            embedding_index[tokens[0]] = [float(t) for t in tokens[1:]]
        embedding_matrix = np.zeros((self.vocab_size, self.embedding_dimension))
        for word, i in self.word_index.items():
            if i >= self.vocab_size:
                continue
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def build_embedding(self):
        """
        build the embedding layer which will be called by the model
        :return: An instance of tensorflow.keras.layers.Embedding
        """
        embeddings_matrix = self.get_embedding_matrix()
        embedding_layer = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dimension,
                                    embeddings_initializer=Constant(embeddings_matrix),
                                    input_length=self.max_length, trainable=True)
        return embedding_layer
