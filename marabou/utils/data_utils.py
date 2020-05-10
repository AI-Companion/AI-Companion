from pathlib import Path
import os
import subprocess


class ImdbDataset:
    """
    Dataset handler which gets training and test data
    """
    def __init__(self, limit=None):
        self.limit = limit
        self.n_obs_per_class = None
        self._get_set()

    def _get_set(self):
        script_path = os.path.join(os.getcwd(), "load_imdb_dataset.sh")
        subprocess.call("%s" % script_path, shell=True)

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

        n_obs = len(os.listdir(os.path.join(directory, 'pos')))
        if self.limit == 0:
            self.limit = 2*n_obs
        else:
            if self.limit > 2*n_obs:
                self.limit = 2*n_obs
        self.n_obs_per_class = round(self.limit / 2)
        for file in Path(os.path.join(directory, 'pos')).iterdir():
            if len(x) >= self.n_obs_per_class:
                break
            x.append(file.read_text())
            y.append(1)
        for file in Path(os.path.join(directory, 'neg')).iterdir():
            if len(x) >= self.limit:
                break
            x.append(file.read_text())
            y.append(0)
        return x, y
