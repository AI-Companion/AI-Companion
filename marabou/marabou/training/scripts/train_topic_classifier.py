import time
from typing import List, Tuple
import os
from itertools import compress
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from dsg.RNN_MTO_classifier import RNNMTO, RNNMTOPreprocessor
from marabou.commons import EMBEDDINGS_DIR, ROOT_DIR, PLOTS_DIR, MODELS_DIR, DATA_DIR, TDConfigReader, TD_CONFIG_FILE


def preprocess_data(X: List, y: List, data_preprocessor: RNNMTOPreprocessor) \
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
    data_preprocessor.fit(X_train, y_train)
    X_train = data_preprocessor.preprocess(X_train)
    X_test = data_preprocessor.preprocess(X_test)
    return X_train, X_test, y_train, y_test


def train_model(config: TDConfigReader) -> None:
    """
    Training function which prints classification summary as as result
    Args:
        config: Configuration object containing parsed .json file parameters
    Return:
        None
    """
    X, y = [], []
    if config.dataset_name == "News20Dataset":
        dataset = fetch_20newsgroups(subset='all', data_home=DATA_DIR, remove=["headers", "footers", "quotes"])
        X = dataset.data
        y = dataset.target
    categories = {
        'rec.sport.baseball': 'sport',
        'rec.sport.hockey': 'sport',
        'rec.autos': 'mechanical',
        'rec.motorcycles': 'mechanical',
        'soc.religion.christian': 'religion',
        'talk.religion.misc': 'religion',
        'talk.politics.guns': 'politics',
        'talk.politics.mideast': 'politics',
        'talk.politics.misc': 'politics',
        'sci.crypt': 'science',
        'sci.electronics': 'science',
        'sci.med': 'science',
        'sci.space': 'science',
        'comp.sys.mac.hardware': 'computer',
        'comp.sys.ibm.pc.hardware': 'computer',
        'comp.os.ms-windows.misc': 'computer',
        'comp.graphics': 'computer'}

    print("===========> Slicing predefined classes from the data")
    label_mapping = {dataset.target_names.index(cat): categories[cat] for cat in categories}
    filter_classes = [cl in label_mapping for cl in y]
    y = list(compress(y, filter_classes))
    X = list(compress(X, filter_classes))
    y = [label_mapping[idx] for idx in y]
    if config.experimental_mode:
        ind = np.random.randint(0, len(X), 3000)
        X = [X[i] for i in ind]
        y = [y[i] for i in ind]
    file_prefix = "topic_classification_%s" % time.strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(MODELS_DIR):
        os.mkdir(EMBEDDINGS_DIR)
    if not os.path.exists(MODELS_DIR):
        os.mkdir(MODELS_DIR)
    if not os.path.exists(PLOTS_DIR):
        os.mkdir(PLOTS_DIR)

    print("===========> Data preprocessing")
    data_preprocessor = RNNMTOPreprocessor(max_sequence_length=config.max_sequence_length,
                                           validation_split=config.validation_split, vocab_size=config.vocab_size)
    X_train, X_test, y_train, y_test = preprocess_data(X, y, data_preprocessor)

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
    print("===========> saving confusion matrix under plots/")
    y_pred = trained_model.predict(X_train)
    labels_list = list(data_preprocessor.labels_to_idx.keys())
    trained_model.save_confusion_matrix(y_pred, y_train, file_prefix, PLOTS_DIR, labels_list)
    print("===========> saving trained model and preprocessor under models/")
    trained_model.save(file_prefix, MODELS_DIR)
    data_preprocessor.save(file_prefix, MODELS_DIR)


def main():
    """main function"""
    if ROOT_DIR is None:
        raise ValueError("please make sure to setup the environment variable MARABOU_HOME\
                         to point for the root of the project")
    train_config = TDConfigReader(TD_CONFIG_FILE)
    train_model(train_config)


if __name__ == '__main__':
    main()
