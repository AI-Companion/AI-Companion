import os
import subprocess
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM
from keras.initializers import Constant
from keras.preprocessing.sequence import pad_sequences
from marabou.utils.config_loader import ConfigReader


class RNNModel:
    """
    Handles the RNN model
    """
    def __init__(self, config: ConfigReader, word_index: Dict):
        self.use_pretrained_embedding = config.pre_trained_embedding
        self.vocab_size = config.vocab_size
        self.embedding_dimension = config.embedding_dimension
        self.embeddings_path = config.embeddings_path
        self.max_length = config.max_sequence_length
        self.word_index = word_index
        self.embedding_layer = self.build_embedding()
        self.model = self.build_model()

    def build_embedding(self):
        """
        builds the embedding layer. depending on the configuration, it will either
        load a pretrained embedding or create an empty embedding to be trained along
        with the data
        :return: None
        """
        if self.use_pretrained_embedding:
            embeddings_matrix = self.get_embedding_matrix()
            embedding_layer = Embedding(self.vocab_size, self.embedding_dimension,
                                        embeddings_initializer=Constant(embeddings_matrix),
                                        input_length=self.max_length, trainable=False)
        else:
            embedding_layer = Embedding(self.vocab_size, self.embedding_dimension, input_length=self.max_length)
        return embedding_layer

    def build_model(self):
        """
        builds an RNN model according to fixed architecture
        :return: None
        """
        print("===========> build model")
        model = Sequential()
        model.add(self.embedding_layer)
        model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(250, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['acc'])
        print(model.summary())
        return model

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

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        """
        fits the model object to the data
        :param X_train: numpy array containing encoded training features
        :param y_train: numpy array containing training targets
        :paran X_test: numpy array containing encoded test features
        :param y_test: numpy array containing test targets
        :return: list of values related to each datasets and loss function
        """
        if (X_test is not None) and (y_test is not None):
            history = self.model.fit(x=X_train, y=y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test),
                                     verbose=2)
        else:
            history = self.model.fit(x=X_train, y=y_train, epochs=10, batch_size=128, verbose=2)
        return history

    def predict(self, text_list, tokenizer_obj):
        """
        inference method
        :param text_list: a list of texts to be evaluated
        :param tokenizer_obj: tokenizer object used to convert the data into training format
        :return: a numpy array containing the probabilities of a positive review for each list entry
        """
        test_samples_tokenized = tokenizer_obj.texts_to_sequences(text_list)
        test_samples_tokenized = pad_sequences(test_samples_tokenized, maxlen=self.max_length)
        return self.model.predict(test_samples_tokenized)

    def save_model(self, file_name_prefix):
        """
        saves the trained model into a h5 file
        :param file_name_prefix: a file name prefix having the following format 'sentiment_analysis_%Y%m%d_%H%M%S'
        :return: None
        """
        model_folder = os.path.join(os.getcwd(), "models")
        if not os.path.isdir(model_folder):
            os.mkdir(model_folder)
        file_url = os.path.join(model_folder, file_name_prefix+"_rnn_model.h5")
        self.model.save(file_url)
        print("----> model saved to %s" % file_url)

    def save_learning_curve(self, history):
        """
        saves the learning curve plot
        :param history: a dictionary object containing training and validation dataset loss function values and
        objective function values for each training iteration
        :return: None
        """
        plot_folder = os.path.join(os.getcwd(), "plots")
        if not os.path.isdir(plot_folder):
            os.mkdir(plot_folder)

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))

        fig, ax = plt.subplots(1, 2)
        ax[0].plot(epochs, acc, 'bo', label='Training acc')
        ax[0].plot(epochs, val_acc, 'b', label='Validation acc')
        ax[0].set_title('Training and validation accuracy')
        ax[0].legend()
        fig.suptitle('model performance')
        ax[1].plot(epochs, loss, 'bo', label='Training loss')
        ax[1].plot(epochs, val_loss, 'b', label='Validation loss')
        ax[1].set_title('Training and validation loss')
        ax[1].legend()
        plt.savefig(os.path.join(plot_folder, "learning_curve.png"))
        plt.close()
