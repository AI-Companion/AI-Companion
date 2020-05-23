import os
import subprocess
import numpy as np
import time
from typing import Dict
from keras.models import Sequential
from keras.layers import GRU, Embedding, Dense, LSTM, Flatten
from keras.initializers import Constant
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
from marabou.utils.config_loader import ConfigReader

class RNNModel:
    def __init__(self, config:ConfigReader, word_index:Dict):
        self.use_pretrained_embedding = config.pre_trained_embedding
        self.vocab_size = config.vocab_size
        self.embedding_dimension = config.embedding_dimension
        self.embeddings_path = config.embeddings_path
        self.max_length = config.max_sequence_length
        self.word_index = word_index
        self.embedding_layer = self.build_embedding()
        self.model = self.build_model()


    def build_embedding(self):
        if self.use_pretrained_embedding:
            embeddings_matrix = self.get_embedding_matrix()
            embedding_layer = Embedding(self.vocab_size, self.embedding_dimension, 
                                        embeddings_initializer=Constant(embeddings_matrix),
                                        input_length=self.max_length, trainable=False)
        else:
            embedding_layer = Embedding(self.vocab_size, self.embedding_dimension, input_length=self.max_length)
        return embedding_layer


    def build_model(self):
        model = Sequential()
        model.add(self.embedding_layer)
        model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(250, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
        print(model.summary())
        return model


    def get_embedding_matrix(self):
        script_path = os.path.join(os.getcwd(), "bash_scripts/load_stanford_6B_embedding.sh")
        subprocess.call("%s %s" % (script_path, self.embeddings_path), shell=True)
        file_name = 'glove.6B.100d.txt'
        if self.embedding_dimension in [50, 100, 200, 300]:
            file_name = 'glove.6B.%id.txt' % self.embedding_dimension
        embedding_index = {}
        with open(os.path.join(os.getcwd(), 'embeddings', file_name)) as f:
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


    def fit(self, X_train, y_train, X_test=None, y_test=None):
        if (X_test is not None) and (y_test is not None):
            history = self.model.fit(x=X_train, y=y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test), verbose =2)
        else:
            history = self.model.fit(x=X_train, y=y_train, epochs=10, batch_size=128, verbose =2)
        return history


    def predict(self, text_list, tokenizer_obj):
        test_samples_tokenized = tokenizer_obj.texts_to_sequences(text_list)
        test_samples_tokenized = pad_sequences(test_samples_tokenized, maxlen=self.max_length)
        self.model.predict(test_samples_tokenized)


    def save_model(self):
        file_name = "rnn_sentiment_analysis_%s" % time.strftime("%Y%m%d_%H%M%S")
        model_folder = os.path.join(os.getcwd(),"models")
        if not os.path.isdir(model_folder):
            os.mkdir(model_folder)
        self.model.save(os.path.join(model_folder, file_name))
