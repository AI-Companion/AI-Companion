import os
import subprocess


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
