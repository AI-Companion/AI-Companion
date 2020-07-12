import time
from typing import List, Tuple
import os
import numpy as np
from src.utils.data_utils import ImdbDataset
from src.utils.config_loader import SentimentAnalysisConfigReader
from src.models.sentiment_analysis_rnn import RNNModel, DataPreprocessor
from src.models.sentiment_analysis_tfidf import DumbModel


def get_training_validation_data(X: List, y: List, data_processor: DataPreprocessor)\
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Wrapper method which yields the training and validation datasets
    Args:
        X: list of texts (features)
        y: list of ratings
        data_processor: a data handler object
    Return:
        tuple containing the training data, validation data
    """
    preprocessed_input = data_processor.clean_data(X)
    preprocessed_input = data_processor.tokenize_text(preprocessed_input)
    X_train, X_test, y_train, y_test = data_processor.split_train_test(preprocessed_input, y)
    return X_train, X_test, y_train, y_test


def train_model(config: SentimentAnalysisConfigReader) -> None:
    """
    Training function which prints classification summary as as result
    Args:
        config: Configuration object containing parsed .json file parameters
    Return:
        None
    """
    X, y = [], []
    if config.dataset_name == "imdb":
        dataset = ImdbDataset(config.dataset_url)
        X, y = dataset.get_set("train")
        X_test, y_test = dataset.get_set("test")
        X = X + X_test
        y = y + y_test
    file_prefix = "sentiment_analysis_%s" % time.strftime("%Y%m%d_%H%M%S")
    if config.model_name == "rnn":
        data_preprocessor = DataPreprocessor(config.max_sequence_length, config.validation_split, config.vocab_size)
        if config.experimental_mode:
            ind = np.random.randint(0, len(X), 1000)
            X = [X[i] for i in ind]
            y = [y[i] for i in ind]
        X_train, X_test, y_train, y_test = get_training_validation_data(X, y, data_preprocessor)
        trained_model = None
        history = []
        trained_model = RNNModel(config=config, data_preprocessor=data_preprocessor)
        history = trained_model.fit(X_train, y_train, X_test, y_test)
        print("===========> saving learning curve under plots/")
        trained_model.save_learning_curve(history, file_prefix)
        print("===========> saving trained model and preprocessor under models/")
        trained_model.save_model(file_prefix)
        data_preprocessor.save_preprocessor(file_prefix)
    else:  # model_name =="tfidf"
        trained_model = DumbModel(config.vocab_size)
        trained_model.fit(X, y)
        print("===========> saving trained model under models")
        trained_model.save_model(file_prefix)


def main():
    """main function"""
    root_dir = os.environ.get("MARABOU_HOME")
    if root_dir is None:
        raise ValueError("please make sure to setup the environment variable MARABOU_ROOT to point for the root of the project")
    config_file_path = os.path.join(root_dir, "marabou/train/config/config_sentiment_analysis.json")
    train_config = SentimentAnalysisConfigReader(config_file_path)
    train_model(train_config)


if __name__ == '__main__':
    main()
