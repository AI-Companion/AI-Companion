from typing import List, Tuple
import time
import os
import numpy as np
from src.utils.data_utils import KaggleDataset
#from src.utils.config_loader import NamedEntityRecognitionConfigReader
#from src.models.named_entity_recognition_rnn import DataPreprocessor, RNNModel
from mlp.named_entity_recognition import NERConfigReader, NERPreprocessor, NERRNN


def get_training_validation_data(X: List, y: List, data_processor: NERPreprocessor)\
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
    #X_train, X_test, y_train, y_test = get_training_validation_data(X, y, data_preprocessor)
    X = data_preprocessor.clean(X)
    X_train, X_test, y_train, y_test = data_preprocessor.split_train_test(X, y)
    data_preprocessor.fit(X_train, y_train)
    X_train, n_tokens_train, y_train = data_preprocessor.preprocess(X_train, y_train)
    X_test, n_tokens_test, y_test = data_preprocessor.preprocess(X_test, y_test)
    print("===========> Model building")
    trained_model = NERRNN(config=config, data_preprocessor=data_preprocessor)
    print(y_test)
    history, report = trained_model.fit(X_train, y_train, X_test, y_test, data_preprocessor.labels_to_idx)
    print("===========> Saving")
    root_dir = os.environ.get("MARABOU_HOME")
    models_dir = os.path.join(root_dir, "marabou/evaluation/trained_models")
    perf_dir = os.path.join(root_dir, "marabou/evaluation/perf")
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    if not os.path.exists(perf_dir):
        os.mkdir(perf_dir)
    print("===========> saving learning curve and classification report under perf/")
    print(history)
    trained_model.save_learning_curve(history, file_prefix, perf_dir)
    trained_model.save_classification_report(report, file_prefix, perf_dir)
    print("===========> saving trained model and preprocessor under models/")
    trained_model.save(file_prefix, models_dir)
    data_preprocessor.save(file_prefix, models_dir)


def main():
    """main function"""
    root_dir = os.environ.get("MARABOU_HOME")
    if root_dir is None:
        raise ValueError("please make sure to setup the environment variable MARABOU_ROOT to point\
                         for the root of the project")
    config_file_path = os.path.join(root_dir, "marabou/train/config/config_named_entity_recognition.json")
    train_config = NERConfigReader(config_file_path)
    train_model(train_config)


if __name__ == '__main__':
    main()
