from typing import List, Tuple
import time
import os
import numpy as np
from mlp.named_entity_recognition import NERConfigReader, NERPreprocessor, NERRNN
from train.utils import KaggleDataset
from commons.definitions import EMBEDDINGS_DIR, NER_CONFIG_FILE, ROOT_DIR, PLOTS_DIR, MODELS_DIR

def preprocess_data(X: List, y: List, data_processor: NERPreprocessor)\
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
    X_train, _ , n_tokens_train, y_train = data_processor.preprocess(X_train, y_train)
    X_test, _, n_tokens_test, y_test = data_processor.preprocess(X_test, y_test)
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
    print("===========> Data preprocessing")
    data_preprocessor = NERPreprocessor(max_sequence_length=config.max_sequence_length,
                                        validation_split=config.validation_split, vocab_size=config.vocab_size)
    X_train, X_test, y_train, y_test = preprocess_data(X, y, data_preprocessor)
    print("===========> Model building")
    trained_model = NERRNN(config=config, data_preprocessor=data_preprocessor, save_folder=EMBEDDINGS_DIR)
    history, report = trained_model.fit(X_train, y_train, X_test, y_test, data_preprocessor.labels_to_idx)
    print("===========> Saving")
    if not os.path.exists(MODELS_DIR):
        os.mkdir(MODELS_DIR)
    if not os.path.exists(PLOTS_DIR):
        os.mkdir(PLOTS_DIR)
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
