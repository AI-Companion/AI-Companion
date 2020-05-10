import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


class DumbModel():
    """
    NB network class. The class provides a wrapper for sklearn methods
    """
    def __init__(self, vocab_size=10_000):
        self.vocab_size = vocab_size
        self.clf = None
        self.vectorizer = None

    def train(self, X_train, y_train):
        """
        training function
        :param X_train: training features
        :param y_train: training targets
        :return: None
        """
        self.vectorizer = TfidfVectorizer(max_features=self.vocab_size)
        X_train = self.vectorizer.fit_transform(X_train)

        self.clf = MultinomialNB()
        self.clf.fit(X_train, y_train)

    def predict_proba(self, X):
        """
        Inference function
        :param X: features to perform inference on
        :return: probability array
        """
        X = self.vectorizer.transform(X)
        y_proba = self.clf.predict_proba(X)
        return y_proba

    def predict(self, X):
        """
        Prediction function
        :param X: features to perform inference on
        :return: prediction label
        """
        X = self.vectorizer.transform(X)
        y_pred = self.clf.predict(X)
        return y_pred

    def serialize(self, file_name):
        """
        saves the model using a pickle serializable
        :param file_name: filename
        :return: None
        """
        with open(file_name, 'wb') as f:
            pickle.dump(self.vocab_size, f)
            pickle.dump(self.vectorizer, f)
            pickle.dump(self.clf, f)

    def get_output(self, probs, query_list):
        """
        gets probability from a given vector of probabilities
        :param probs: probability vector for the given query
        :query_list: a list of the strings given to the model to predict
        :return: a dict containing probability prediction for each queried string
        """
        vals = [round(p[1], 4) for p in probs]
        keys = ["class1_probs_%s" % name for name in query_list]
        out = dict.fromkeys(keys)
        out.update(zip(keys, vals))
        return out

    @staticmethod
    def deserialize(file_name):
        """
        extracts a model saved using the serialize function
        :param file_name: name of the file containing the saved model
        :return: a model object
        """
        model = DumbModel()
        with open(file_name, 'rb') as f:
            model.vocab_size = pickle.load(f)
            model.vectorizer = pickle.load(f)
            model.clf = pickle.load(f)
            return model
