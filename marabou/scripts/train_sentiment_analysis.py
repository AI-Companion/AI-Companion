import os
import argparse
from typing import List, Tuple
import numpy as np
from sklearn.metrics import classification_report
from keras.preprocessing.text import Tokenizer
from marabou.utils.data_utils import ImdbDataset, DataPreprocessor
from marabou.utils.config_loader import ConfigReader
from marabou.models.sentiment_analysis.rnn_models import RNNModel


def get_training_validation_data(X: List, y: List, data_processor:DataPreprocessor)\
    -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tokenizer]:
    preprocessed_input = data_processor.clean_data(X)
    tokenizer_obj, preprocessed_input = data_processor.tokenize_text(preprocessed_input)
    X_train, X_test, y_train, y_test = data_processor.split_train_test(preprocessed_input, y)
    return X_train, X_test, y_train, y_test, tokenizer_obj

def train_model(config: ConfigReader) -> None:
    """
    training function which prints classification summary as as result
    :param config: Configuration object containing parsed .json file parameters
    :return: None
    """
    X, y = [], []
    if config.dataset_name == "imdb":
        dataset = ImdbDataset(config.dataset_url)
        X, y = dataset.get_set("train")
        X_test, y_test = dataset.get_set("train")
        X = X + X_test
        y = y + y_test
    data_preprocessor = DataPreprocessor(config.max_sequence_length, config.validation_split, config.vocab_size)
    X_train, X_test, y_train, y_test, tokenizer_obj = get_training_validation_data(X, y, data_preprocessor)
    trained_model = RNNModel(config, tokenizer_obj.word_index)
    trained_model.fit()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train sentiment analysis model")
    parser.add_argument('--config','-c', help='Path to the configuration file', required=True)

    return parser.parse_args()

def main():
    """main function"""
    args = parse_arguments()
    train_config = ConfigReader(args.config)
    train_model(train_config)

if __name__ == '__main__':
    main()
