import time
from typing import List, Tuple
import os
import numpy as np
from src.utils.data_utils import ImdbDataset
from mlp.sentiment_analysis import SAConfigReader, SAPreprocessor, SARNN

def get_training_validation_data(X: List, y: List, data_processor: SAPreprocessor)\
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


def train_model(config: SAConfigReader) -> None:
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
    if config.experimental_mode:
        ind = np.random.randint(0, len(X), 1000)
        X = [X[i] for i in ind]
        y = [y[i] for i in ind]
    file_prefix = "sentiment_analysis_%s" % time.strftime("%Y%m%d_%H%M%S")
    print("===========> Data preprocessing")
    data_preprocessor = SAPreprocessor(max_sequence_length=config.max_sequence_length,\
                                       validation_split=config.validation_split, vocab_size=config.vocab_size)
    X = data_preprocessor.clean(X)
    X_train, X_test, y_train, y_test = data_preprocessor.split_train_test(X, y)
    data_preprocessor.fit(X_train)
    X_train = data_preprocessor.preprocess(X_train)
    X_test = data_preprocessor.preprocess(X_test)
    print("===========> Model building")
    trained_model = SARNN(config=config, data_preprocessor=data_preprocessor)
    history = trained_model.fit(X_train, y_train, X_test, y_test)
    print("===========> saving")
    root_dir = os.environ.get("MARABOU_HOME")
    models_dir = os.path.join(root_dir, "marabou/evaluation/trained_models")
    plots_dir = os.path.join(root_dir, "marabou/evaluation/plots")
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
    print("===========> saving learning curve under plots/")
    trained_model.save_learning_curve(history, file_prefix, plots_dir)
    print("===========> saving trained model and preprocessor under models/")
    trained_model.save(file_prefix, models_dir)
    data_preprocessor.save(file_prefix, models_dir)


def main():
    """main function"""
    root_dir = os.environ.get("MARABOU_HOME")
    if root_dir is None:
        raise ValueError("please make sure to setup the environment variable MARABOU_ROOT to point for the root of the project")
    config_file_path = os.path.join(root_dir, "marabou/train/config/config_sentiment_analysis.json")
    train_config = SAConfigReader(config_file_path)
    train_model(train_config)


if __name__ == '__main__':
    main()
