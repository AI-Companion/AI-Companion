from typing import List, Tuple
import time
import os
import numpy as np
from dsg.RNN_MTM_classifier import RNNMTMPreprocessor, RNNMTM 
from marabou.training.datasets import KaggleDataset
from marabou.commons import EMBEDDINGS_DIR, NER_CONFIG_FILE, ROOT_DIR, PLOTS_DIR, MODELS_DIR, NERConfigReader

def preprocess_data(X: List, y: List, data_processor: RNNMTMPreprocessor)\
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
    X = data_processor.clean(X)
    X_train, X_test, y_train, y_test = data_processor.split_train_test(X, y)
    data_processor.fit(X_train, y_train)
    X_train, _, _, y_train = data_processor.preprocess(X_train, y_train)
    X_test, _, _, y_test = data_processor.preprocess(X_test, y_test)
    return X_train, X_test, y_train, y_test


def train_model(config: NERConfigReader) -> None:
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
    if config.experimental_mode:
        ind = np.random.randint(0, len(X), 500)
        X = [X[i] for i in ind]
        y = [y[i] for i in ind]
    file_prefix = "named_entity_recognition_%s" % time.strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(MODELS_DIR):
        os.mkdir(EMBEDDINGS_DIR)
    if not os.path.exists(MODELS_DIR):
        os.mkdir(MODELS_DIR)
    if not os.path.exists(PLOTS_DIR):
        os.mkdir(PLOTS_DIR)
    print("===========> Data preprocessing")
    data_preprocessor = RNNMTMPreprocessor(max_sequence_length=config.max_sequence_length,
                                        validation_split=config.validation_split, vocab_size=config.vocab_size)
    X_train, X_test, y_train, y_test = preprocess_data(X, y, data_preprocessor)
    print("===========> Model building")
    trained_model = RNNMTM(pre_trained_embedding=config.pre_trained_embedding,
                           vocab_size=config.vocab_size,
                           embedding_dimension=config.embedding_dimension,
                           embedding_algorithm=config.embedding_algorithm,
                           n_iter=config.n_iter,
                           embeddings_path=config.embeddings_path,
                           max_sequence_length=config.max_sequence_length,
                           data_preprocessor=data_preprocessor, save_folder=EMBEDDINGS_DIR)
    history, report = trained_model.fit(X_train, y_train, X_test, y_test, data_preprocessor.labels_to_idx)
    print("===========> Saving")
    print("===========> saving learning curve and classification report under perf/")
    print(history)
    trained_model.save_learning_curve(history, file_prefix, PLOTS_DIR, metric="crf_viterbi_accuracy")
    trained_model.save_classification_report(report, file_prefix, PLOTS_DIR)
    print("===========> saving trained model and preprocessor under models/")
    trained_model.save(file_prefix, MODELS_DIR)
    data_preprocessor.save(file_prefix, MODELS_DIR)


def main():
    """main function"""
    if ROOT_DIR is None:
        raise ValueError("please make sure to setup the environment variable MARABOU_ROOT to point\
                         for the root of the project")
    train_config = NERConfigReader(NER_CONFIG_FILE)
    train_model(train_config)


if __name__ == '__main__':
    main()
