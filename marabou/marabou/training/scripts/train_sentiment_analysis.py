import time
from typing import List, Tuple, Dict
import os
import numpy as np
from marabou.training.datasets import ImdbDataset
from dsg.RNN_MTO_classifier import RNNMTO, RNNMTOPreprocessor
from marabou.commons import ROOT_DIR, PLOTS_DIR, MODELS_DIR, SA_CONFIG_FILE, SAConfigReader, EMBEDDINGS_DIR


def preprocess_data(X: List, y: List, data_preprocessor: RNNMTOPreprocessor, labels_to_idx:Dict=None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Wrapper method which yields the training and validation datasets
    Args:
        X: list of unprocessed features
        y: list of ratings
        data_processor: a data handler object
    Return:
        tuple containing the training data, validation data
    """
    X = data_preprocessor.clean(X)
    X_train, X_test, y_train, y_test = data_preprocessor.split_train_test(X, y)
    data_preprocessor.fit(X_train, y, labels_to_idx=labels_to_idx)
    X_train = data_preprocessor.preprocess(X_train)
    X_test = data_preprocessor.preprocess(X_test)
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
    
    labels_to_idx = {"pos":1, "neg":0} # mapping the labels to corresponding indices
    file_prefix = "sentiment_analysis_%s" % time.strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(MODELS_DIR):
        os.mkdir(EMBEDDINGS_DIR)
    if not os.path.exists(MODELS_DIR):
        os.mkdir(MODELS_DIR)
    if not os.path.exists(PLOTS_DIR):
        os.mkdir(PLOTS_DIR)
    print("===========> Data preprocessing")
    data_preprocessor = RNNMTOPreprocessor(max_sequence_length=config.max_sequence_length, \
                                           validation_split=config.validation_split, vocab_size=config.vocab_size)
    X_train, X_test, y_train, y_test = preprocess_data(X, y, data_preprocessor, labels_to_idx)
    print("===========> Model building")
    trained_model = RNNMTO(pre_trained_embedding=config.pre_trained_embedding,
                           vocab_size=config.vocab_size,
                           embedding_dimension=config.embedding_dimension,
                           embedding_algorithm=config.embedding_algorithm,
                           n_iter=config.n_iter,
                           embeddings_path=config.embeddings_path,
                           max_sequence_length=config.max_sequence_length,
                           data_preprocessor=data_preprocessor, save_folder=EMBEDDINGS_DIR)
    history = trained_model.fit(X_train, y_train, X_test, y_test)
    print("===========> saving")
    print("===========> saving learning curve under plots/")
    trained_model.save_learning_curve(history, file_prefix, PLOTS_DIR)
    print("===========> saving trained model and preprocessor under models/")
    trained_model.save(file_prefix, MODELS_DIR)
    data_preprocessor.save(file_prefix, MODELS_DIR)


def main():
    """main function"""
    if ROOT_DIR is None:
        raise ValueError(
            "please make sure to setup the environment variable MARABOU_ROOT to point for the root of the project")
    train_config = SAConfigReader(SA_CONFIG_FILE)
    train_model(train_config)


if __name__ == '__main__':
    main()
