from pathlib import Path
import os
import subprocess


class SentimentAnalysisDataset:
    """
    Dataset handler which gets training and test data
    """
    def __init__(self, limit=None):
        self.limit = limit
        self._get_set()

    def _get_set(self):
        script_path = os.path.join(os.getcwd(), "load_sentiment_analysis_data.sh")
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

        for file in Path(os.path.join(directory, 'pos')).iterdir():
            if self.limit is not None and len(x) >= self.limit / 2:
                break
            x.append(file.read_text())
            y.append(1)
        for file in Path(os.path.join(directory, 'neg')).iterdir():
            if self.limit is not None and len(x) >= self.limit:
                break
            x.append(file.read_text())
            y.append(0)
        return x, y
