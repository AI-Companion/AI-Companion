from typing import List, Tuple
import time
import numpy as np
from marabou.utils.data_utils import KaggleDataset
from marabou.utils.config_loader import NamedEntityRecognitionConfigReader
from marabou.models.named_entity_recognition_rnn import DataPreprocessor, RNNModel


def get_training_validation_data(X: List, y: List, data_processor: DataPreprocessor)\
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    wrapper method which yields the training and validation datasets
    :param X: list of texts (features)
    :param y: list of ratings
    :param data_processor: a data handler object
    :return: tuple containing the training data, validation data
    """
    preprocessed_input = data_processor.clean_data(X)
    preprocessed_input, y = data_processor.tokenize_text(preprocessed_input, y)
    X_train, X_test, y_train, y_test = data_processor.split_train_test(preprocessed_input, y)
    return X_train, X_test, y_train, y_test


def train_model(config: NamedEntityRecognitionConfigReader) -> None:
    """
    training function which prints classification summary as as result
    :param config: Configuration object containing parsed .json file parameters
    :return: None
    """
    X, y = [], []
    if config.dataset_name == "kaggle_ner":
        dataset = KaggleDataset(config.dataset_url)
        X, y = dataset.get_set()
    data_preprocessor = DataPreprocessor(config.max_sequence_length, config.validation_split, config.vocab_size)
    if config.experimental_mode:
        ind = np.random.randint(0, len(X), 5000)
        X = [X[i] for i in ind]
        y = [y[i] for i in ind]
    X_train, X_test, y_train, y_test = get_training_validation_data(X, y, data_preprocessor)
    file_prefix = "sentiment_analysis_%s" % time.strftime("%Y%m%d_%H%M%S")
    trained_model = RNNModel(config=config, data_preprocessor=data_preprocessor)
    history = trained_model.fit(X_train, y_train, X_test, y_test)
    trained_model.save_learning_curve(history, file_prefix)


def main():
    """main function"""
    train_config = NamedEntityRecognitionConfigReader("config/config_named_entity_recognition.json")
    train_model(train_config)


if __name__ == '__main__':
    main()
