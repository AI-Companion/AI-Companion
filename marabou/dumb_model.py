import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


class DumbModel:
    def __init__(self, vocab_size=10_000):
        self.vocab_size = vocab_size
        self.clf = None
        self.vectorizer = None

    def train(self, X_train, y_train):
        self.vectorizer = TfidfVectorizer(max_features=self.vocab_size)
        X_train = self.vectorizer.fit_transform(X_train)

        self.clf = MultinomialNB()
        self.clf.fit(X_train, y_train)

    def predict_proba(self, X):
        X = self.vectorizer.transform(X)
        y_proba = self.clf.predict_proba(X)
        return y_proba

    def predict(self, X):
        X = self.vectorizer.transform(X)
        y_pred = self.clf.predict(X)
        return y_pred

    def serialize(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self.vocab_size, f)
            pickle.dump(self.vectorizer, f)
            pickle.dump(self.clf, f)
    
    @staticmethod
    def deserialize(fname):
        model = DumbModel()
        with open(fname, 'rb') as f:
            model.vocab_size = pickle.load(f)
            model.vectorizer = pickle.load(f)
            model.clf = pickle.load(f)
            
            return model