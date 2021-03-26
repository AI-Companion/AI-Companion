import time
import os
import numpy as np
from dsg.CNN_classifier import CNNClassifier, CNNClassifierPreprocessor
from marabou.training.datasets import FashionImageNet
from marabou.commons import CCConfigReader, MODELS_DIR, PLOTS_DIR, ROOT_DIR, CC_CONFIG_FILE


def train_model(config: CCConfigReader) -> None:
    """
    training function which prints classification summary as as result
    Args:
        config: Configuration object containing parsed .json file parameters
    Return:
        None
    """
    X, y = [], []
    if config.dataset_name == "fashion_mnist":
        dataset = FashionImageNet()
        X, y = dataset.get_set()
    if config.experimental_mode:
        ind = np.random.randint(0, len(X), 50)
        X = [X[i] for i in ind]
        y = [y[i] for i in ind]
    file_prefix = "clothing_classifier_%s" % time.strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(MODELS_DIR):
        os.mkdir(MODELS_DIR)
    if not os.path.exists(PLOTS_DIR):
        os.mkdir(PLOTS_DIR)

    preprocessor = CNNClassifierPreprocessor(image_height=config.image_height,
                                             image_width=config.image_width, validation_split=config.validation_split)
    X_train, X_test, y_train, y_test = preprocessor.split_train_test(X, y)
    preprocessor.fit(X_train, y_train)
    X_train, y_train = preprocessor.preprocess(X_train, y_train)
    X_test, y_test = preprocessor.preprocess(X_test, y_test)
    file_prefix = "fashion_classifier_%s" % time.strftime("%Y%m%d_%H%M%S")
    idx_to_labels = {v: k for k, v in preprocessor.labels_to_idx.items()}
    trained_model = CNNClassifier(idx_to_labels=idx_to_labels,
                                  pre_trained_cnn=config.use_pre_trained_cnn,
                                  pretrained_network_name=config.pretrained_network_name,
                                  n_iter=config.n_iter,
                                  image_height=config.image_height,
                                  image_width=config.image_width,
                                  batch_size=config.batch_size,
                                  pretrained_network_path=config.pretrained_network_path)
    history, report = trained_model.fit(X_train, y_train, X_test, y_test)
    print("===========> Saving")
    print("===========> saving learning curve and classification report under perf/")
    trained_model.save_learning_curve(history, file_prefix, PLOTS_DIR)
    trained_model.save_classification_report(report, file_prefix, PLOTS_DIR)
    print("===========> saving trained model and preprocessor under models/")
    trained_model.save(file_prefix, MODELS_DIR)
    preprocessor.save(file_prefix, MODELS_DIR)


def main():
    """main function"""
    if ROOT_DIR is None:
        raise ValueError("please make sure to setup the environment variable MARABOU_ROOT to point\
                         for the root of the project")
    train_config = CCConfigReader(CC_CONFIG_FILE)
    train_model(train_config)


if __name__ == '__main__':
    main()
