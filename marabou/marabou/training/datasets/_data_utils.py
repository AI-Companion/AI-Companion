import os
import subprocess
from cv2 import cv2
import pandas as pd
import wget
import tarfile
import gdown
from zipfile import ZipFile, BadZipfile
from marabou.commons import SCRIPTS_DIR, DATA_DIR


class ImdbDataset:
    """
    Dataset handler for the imdb dataset
    """

    def __init__(self, dataset_url=None):
        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)
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
        output_file_name = os.path.join(DATA_DIR, "imdb.tar.gz")
        url = 'https://drive.google.com/uc?id={}'.format(dataset_url)
        if not os.path.isfile(output_file_name):
            print("---> Collecting dataset")
            gdown.download(url, output_file_name, quiet=True)
        else:
            print("---> Dataset already downloaded")
        tar = tarfile.open(output_file_name)
        tar.extractall(path=DATA_DIR)
        tar.close()

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
        directory = os.path.join(DATA_DIR, "aclImdb")
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
            y.append("pos")
        for f in os.listdir(os.path.join(directory, 'neg')):
            file1 = open(os.path.join(directory, 'neg', f), "r")
            x.append(file1.readline())
            file1.close()
            y.append("neg")
        return x, y


class News20Dataset:
    """
    Dataset handler for the imdb dataset
    """

    def __init__(self, dataset_url=None):
        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)
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
        output_file_name = os.path.join(DATA_DIR, "20news-18828.tar.gz")
        if not os.path.isfile(output_file_name):
            print("---> Collecting dataset")
            output_file_name = wget.download(dataset_url, out=output_file_name)
            print("")
        else:
            print("---> Dataset already downloaded")
        tar = tarfile.open(output_file_name)
        tar.extractall(path=DATA_DIR)
        tar.close()

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
        directory = os.path.join(DATA_DIR, "aclImdb")
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
        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)
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
        output_file_name = os.path.join(DATA_DIR, "ner_dataset.zip")
        url = 'https://drive.google.com/uc?id={}'.format(dataset_url)
        output_file_name = os.path.join(DATA_DIR, "ner_dataset.zip")
        if not os.path.isfile(output_file_name):
            print("---> Collecting dataset")
            gdown.download(url, output_file_name, quiet=True)
        else:
            print("---> Dataset already downloaded")
        with ZipFile(output_file_name, 'r') as zipObj:
            zipObj.extractall(DATA_DIR)

    def get_set(self):
        """
        Retrieves the compressed data file and extracts the training data.
        Then it deletes the extracted csv
        Return:
            training features and targets
        """
        X = []
        y = []
        dataset_path = os.path.join(DATA_DIR, "ner_dataset.csv")
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


class FashionImageNet:
    """
    Dataset handler for the fashion imagenet dataset
    """

    def __init__(self, dataset_url=None):
        self.dataset_url = dataset_url

    def get_set(self):
        """
        Retrieves fashion imagenet from zipped file on the disk
        Return:
            features, targets
        """
        print("===========> extracting fashion imagenet dataset")
        data_folder = os.path.join(DATA_DIR, "clothing_classifier")
        subfolders_list = os.listdir(data_folder)
        subfolders_list = [f for f in subfolders_list if os.path.isdir(os.path.join(data_folder, f))]
        X = []
        y = []
        for folder in subfolders_list:
            class_folder = os.path.join(data_folder, folder)
            files_list = [f for f in os.listdir(class_folder) if os.path.isfile(os.path.join(class_folder, f))]
            files_list.remove("status.txt")
            for f in files_list:
                file_url = os.path.join(class_folder, f)
                im = cv2.imread(file_url)
                if im is not None:
                    X.append(file_url)
                    y.append(folder)
        return X, y
