from typing import List, Tuple
import time
import os
import numpy as np
from src.utils.data_utils import KaggleDataset
from src.utils.config_loader import NamedEntityRecognitionConfigReader
from src.models.named_entity_recognition_rnn import DataPreprocessor, RNNModel


def get_training_validation_data(X: List, y: List, data_processor: DataPreprocessor)\
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    wrapper method which yields the training and validation datasets
    Args:
        X: list of texts (features)
        y: list of ratings
        data_processor: a data handler object
    Return:
        tuple containing the training data, validation data
    """
    preprocessed_input = data_processor.clean_data(X)
    preprocessed_input, y = data_processor.tokenize_text(preprocessed_input, y)
    X_train, X_test, y_train, y_test = data_processor.split_train_test(preprocessed_input, y)
    return X_train, X_test, y_train, y_test


def train_model(config: NamedEntityRecognitionConfigReader) -> None:
    """
    training function which prints classification summary as as result
    Args:
        config: Configuration object containing parsed .json file parameters
    Return:
        None
    """
    X, y = [], []
    if config.dataset_name == "kaggle_ner":
        dataset = KaggleDataset(config.dataset_url)
        X, y = dataset.get_set()
    data_preprocessor = DataPreprocessor(config.max_sequence_length, config.validation_split, config.vocab_size)
    if config.experimental_mode:
        ind = np.random.randint(0, len(X), 1000)
        X = [X[i] for i in ind]
        y = [y[i] for i in ind]
    X_train, X_test, y_train, y_test = get_training_validation_data(X, y, data_preprocessor)

    file_prefix = "named_entity_recognition_%s" % time.strftime("%Y%m%d_%H%M%S")
    trained_model = RNNModel(config=config, data_preprocessor=data_preprocessor)
    history, report = trained_model.fit(X_train, y_train, X_test, y_test, data_preprocessor.labels_to_idx)
    print("===========> saving learning curve and classification report under perf/")
    trained_model.save_learning_curve(history, file_prefix)
    trained_model.save_classification_report(report, file_prefix)
    print("===========> saving trained model and preprocessor under models/")
    trained_model.save_model(file_prefix)
    data_preprocessor.save_preprocessor(file_prefix)


def main():
    """main function"""
    root_dir = os.environ.get("MARABOU_HOME")
    config_file_path = os.path.join(root_dir, "train/config/config_named_entity_recognition.json")
    train_config = NamedEntityRecognitionConfigReader(config_file_path)
    train_model(train_config)


if __name__ == '__main__':
    main()
