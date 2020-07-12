import pickle
import os
import re
from itertools import compress
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


class DumbModel():
    """
    NB network class. The class provides a wrapper for sklearn methods
    """
    def __init__(self, vocab_size=10_000):
        self.model_name = "tfidf_model"
        self.vocab_size = vocab_size
        self.clf = None
        self.vectorizer = None

    def fit(self, X_train, y_train):
        """
        Training function
        Args:
            X_train: training features
            y_train: training targets
        Return:
            None
        """
        self.vectorizer = TfidfVectorizer(max_features=self.vocab_size)
        X_train = self.vectorizer.fit_transform(X_train)

        self.clf = MultinomialNB()
        self.clf.fit(X_train, y_train)

    def predict_proba(self, X):
        """
        Inference function
        Args:
            X: features to perform inference on
        Return:
            probability array
        """
        X = self.vectorizer.transform(X)
        probs = self.clf.predict_proba(X)

        return [p[1] for p in probs]

    def predict(self, X):
        """
        Prediction function
        Args:
            X: features to perform inference on
        Return:
            prediction label
        """
        X = self.vectorizer.transform(X)
        y_pred = self.clf.predict(X)
        return y_pred

    def save_model(self, file_name_prefix):
        """
        Saves the model using a pickle serializable
        Args:
            file_name_prefix: a file name prefix having the following format 'sentiment_analysis_%Y%m%d_%H%M%S'
        Return:
            None
        """
        model_folder = os.path.join(os.getcwd(), "models")
        if not os.path.isdir(model_folder):
            os.mkdir(model_folder)
        file_url = os.path.join(model_folder, file_name_prefix + "_tfidf_model.pickle")
        with open(file_url, 'wb') as f:
            pickle.dump(self.vocab_size, f)
            pickle.dump(self.vectorizer, f)
            pickle.dump(self.clf, f)

    def get_output(self, probs, query_list):
        """
        Gets probability from a given vector of probabilities
        Args:
            probs: probability vector for the given query
            query_list: a list of the strings given to the model to predict
        Return:
            dict containing probability prediction for each queried string
        """
        vals = [round(p[1], 4) for p in probs]
        keys = ["class1_probs_%s" % name for name in query_list]
        out = dict.fromkeys(keys)
        out.update(zip(keys, vals))
        return out

    @staticmethod
    def load_model():
        """
        Finds the most recent saved model and loads it, otherwise returns None object
        Return:
            model object
        """
        model = None
        model_dir = os.path.join(os.getcwd(), "models")
        model_files_list = os.listdir(model_dir)
        if len(model_files_list) > 0:
            tfidf_models_idx = [("sentiment_analysis" in f) and ("tfidf" in f) for f in model_files_list]
            if sum(tfidf_models_idx) > 0:
                tfidf_model = list(compress(model_files_list, tfidf_models_idx))
                model_dates = [int(''.join(re.findall(r'\d+', f))) for f in tfidf_model]
                model = DumbModel()
                file_name = tfidf_model[np.argmax(model_dates)]
                with open(os.path.join(model_dir, file_name), 'rb') as f:
                    model.vocab_size = pickle.load(f)
                    model.vectorizer = pickle.load(f)
                    model.clf = pickle.load(f)
                return model
            return None
        return None
