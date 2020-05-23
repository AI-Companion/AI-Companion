import os
import subprocess
from typing import List
import string
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')


class ImdbDataset:
    """
    Dataset handler which gets training and test data
    """
    def __init__(self, dataset_url=None):
        self._get_set(dataset_url)

    def _get_set(self, dataset_url):
        """
        checks if imdb dataset is downloaded or not, if not it'll collect it
        :param dataset_url: web address of the imdb dataset
        :return: None
        """
        print("===========> imdb dataset collection")
        script_path = os.path.join(os.getcwd(), "bash_scripts/load_imdb_dataset.sh")
        subprocess.call("%s %s" % (script_path, dataset_url), shell=True)

    def get_set(self, mode="train"):
        """
        returns training data with the given number of rows
        :param limit: max number of rows
        :return: training features and targets
        """
        x = []
        y = []
        directory = os.path.join(os.getcwd(), "data/aclImdb")
        if mode == "train":
            directory = os.path.join(directory, "train")
        else:
            directory = os.path.join(directory, "test")

        n_obs = 2 * len(os.listdir(os.path.join(directory, 'pos')))
        n_obs_per_class = round(n_obs / 2)
        for f in os.listdir(os.path.join(directory, 'pos')):
            if len(x) >= n_obs_per_class:
                break
            file1 = open(os.path.join(directory, 'pos', f), "r")
            x.append(file1.readline())
            file1.close()
            y.append(1)
        for f in os.listdir(os.path.join(directory, 'neg')):
            file1 = open(os.path.join(directory, 'neg', f), "r")
            x.append(file1.readline())
            file1.close()
            y.append(0)
        return x, y


class DataPreprocessor:
    """
    Utility class performing several data preprocessing steps
    """
    def __init__(self, max_sequence_length: int, validation_split: float, vocab_size: int):
        self.max_sequence_length = max_sequence_length
        self.validation_split = validation_split
        self.vocab_size = vocab_size
        self.tokenizer_obj = None

    def clean_data(self, X: List):
        """
        performs data cleaning operations such as removing html breaks, lower case,
        remove stopwords ...
        :param X: input reviews to be cleaned
        :return: None
        """
        print("===========> data cleaning")
        review_lines = list()
        for line in X:
            line = line.replace('<br /><br />', ' ')
            line = line.replace('<br />', ' ')
            tokens = word_tokenize(line)
            tokens = [w.lower() for w in tokens]
            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in tokens]
            words = [word for word in stripped if word.isalpha()]
            stop_words = set(stopwords.words('english'))
            words = [word for word in words if word not in stop_words]
            review_lines.append(words)
        print("----> data cleaning finish")
        return review_lines

    def tokenize_text(self, X: List):
        """
        performs data tokenization into a format that is digestible by the model
        :param X: list of predictors already cleaned
        :return: tokenizer object and tokenized input features
        """
        print("===========> data tokenization")
        tokenizer_obj = Tokenizer(num_words=self.vocab_size)
        tokenizer_obj.fit_on_texts(X)
        self.tokenizer_obj = tokenizer_obj
        sequences = tokenizer_obj.texts_to_sequences(X)
        word_index = tokenizer_obj.word_index
        review_pad = pad_sequences(sequences, maxlen=self.max_sequence_length)
        print("----> data tokenization finish")
        print("found %i unique tokens" % len(word_index))
        print("features tensor shape ", review_pad.shape)
        return tokenizer_obj, review_pad

    def split_train_test(self, X, y):
        """
        wrapper method to split training data into a validation set and a training set
        :param X: tokenized predictors
        :param y: labels
        :return: a tuple consisting of training predictors, training labels, validation predictors, validation labels
        """
        print("===========> data split")
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=self.validation_split)
        print("----> data split finish")
        print('training features shape ', X_train.shape)
        print('testing features shape ', X_test.shape)
        print('training target shape ', np.asarray(y_train).shape)
        print('testing target shape ', np.asarray(y_test).shape)
        return X_train, X_test, np.asarray(y_train), np.asarray(y_test)

    def save_tokenizer(self, file_name_prefix):
        """
        stores the tokenizer under 'models folder'
        :param file_name_prefix: a file name prefix having the following format 'sentiment_analysis_%Y%m%d_%H%M%S'
        :return: None
        """
        model_folder = os.path.join(os.getcwd(), "models")
        if not os.path.isdir(model_folder):
            os.mkdir(model_folder)
        file_url = os.path.join(model_folder, file_name_prefix+"_tokenizer.pickle")
        with open(file_url, 'wb') as handle:
            pickle.dump(self.tokenizer_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("----> tokenizer object saved to %s" % file_url)
