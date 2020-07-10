import time
import os
import numpy as np
from src.utils.data_utils import FashionImageNet
from src.utils.config_loader import FashionClassifierConfigReader
from src.models.cnn_classifier import DataPreprocessor, CNNClothing


def train_model(config: FashionClassifierConfigReader) -> None:
    """
    training function which prints classification summary as as result
    Args:
        config: Configuration object containing parsed .json file parameters
    Return:
        None
    """
    X, y = [], []
    if config.dataset_name == "fashion_mnist":
        dataset = FashionImageNet(config.dataset_url)
        X, y = dataset.get_set()
    if X is None or y is None:
        raise ValueError("please make sure you have the correct dataset link in the config file")
    if config.experimental_mode:
        ind = np.random.randint(0, len(X), 500)
        X = [X[i] for i in ind]
        y = [y[i] for i in ind]
    preprocessor = DataPreprocessor(config)
    X = preprocessor.load_images(X)
    X_train, X_test, y_train, y_test, idx_to_labels = preprocessor.split_train_test(X, y)

    file_prefix = "fashion_imagenet_%s" % time.strftime("%Y%m%d_%H%M%S")
    trained_model = CNNClothing(idx_to_labels, config=config)
    history, report = trained_model.fit(X_train, y_train, X_test, y_test)
    print("===========> saving learning curve and classification report under perf/")
    trained_model.save_learning_curve(history, file_prefix)
    trained_model.save_classification_report(report, file_prefix)
    print("===========> saving trained model and preprocessor under models/")
    trained_model.save_model(file_prefix)


def main():
    """main function"""
    root_dir = os.environ.get("MARABOU_HOME")
    config_file_path = os.path.join(root_dir, "train/config/config_fashion_classifier.json")
    train_config = FashionClassifierConfigReader(config_file_path)
    train_model(train_config)


if __name__ == '__main__':
    main()
