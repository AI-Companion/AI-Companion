import os
import subprocess
import pandas as pd


class ImdbDataset:
    """
    Dataset handler for the imdb dataset
    """
    def __init__(self, dataset_url=None):
        self._get_set(dataset_url)

    def _get_set(self, dataset_url):
        """
        Checks if imdb dataset is downloaded or not, if not it'll collect it
        Args:
            dataset_url: web address of the imdb dataset
        Return:
            None
        """
        print("===========> imdb dataset collection")
        script_path = os.path.join(os.getcwd(), "bash_scripts/load_imdb_dataset.sh")
        subprocess.call("%s %s" % (script_path, dataset_url), shell=True)

    def get_set(self, mode="train"):
        """
        Returns training data with the given number of rows
        Args:
            limit: max number of rows
        Return:
            training features and targets
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


class KaggleDataset:
    """
    Dataset handler for the ner dataset
    """
    def __init__(self, dataset_url=None):
        self._get_set(dataset_url)

    def _get_set(self, dataset_url):
        """
        Checks if imdb dataset is downloaded or not, if not it'll collect it
        Args:
            dataset_url: web address of the imdb dataset
        Return:
            None
        """
        print("===========> extracting kaggle ner dataset")
        script_path = os.path.join(os.getcwd(), "bash_scripts/load_ner_dataset.sh")
        subprocess.call("%s %s" % (script_path, dataset_url), shell=True)

    def get_set(self):
        """
        Retrieves the compressed data file and extracts the training data.
        Then it deletes the extracted csv
        Return:
            training features and targets
        """
        X = []
        y = []
        dataset_path = os.path.join(os.getcwd(), "data/ner_dataset.csv")
        data = pd.read_csv(dataset_path, encoding="latin1")
        data = data.fillna(method="ffill")
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        sent_grouped = data.groupby("Sentence #").apply(agg_func)
        sent_list = [s for s in sent_grouped]
        X = [[w[0] for w in s] for s in sent_list]
        y = [[w[2] for w in s] for s in sent_list]
        os.remove(dataset_path)
        return X, y
