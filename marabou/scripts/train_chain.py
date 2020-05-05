import os
import argparse
from sklearn.metrics import classification_report
from marabou.dataset import Dataset
from marabou.models.dumb_model import DumbModel


def train_model(dataset_dir: str, model_file_name: str, vocab_size: int) -> None:
    """
    training function which prints classification summary as as result
    :param dataset_dir: relative path for dataset folder
    :param model_file_url: desired path to save model file
    :param vocab_size: number of rows to read from the training dataset
    :return: None
    """
    print(f'Training model from directory {dataset_dir}')
    print(f'Vocabulary size: {vocab_size}')

    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    dset = Dataset(train_dir, test_dir)
    X, y = dset.get_train_set()

    model = DumbModel(vocab_size=vocab_size)
    model.train(X, y)

    print(f'Storing model to {model_file_name}')
    model.serialize(model_file_name)

    X_test, y_test = dset.get_test_set()
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))


def parse_arguments():
    """script arguments parser"""
    parser = argparse.ArgumentParser(description="Train sentiment analysis classifier")
    parser.add_argument('model_file', help='model file', type=str)
    parser.add_argument('dataset_dir', help='dataset directory', type=str)
    parser.add_argument('--vocab_size', help='volcabulary size', type=int)

    return parser.parse_args()


def main():
    """main function"""
    args = parse_arguments()

    train_model(args.dataset_dir, args.model_file, int(args.vocab_size))


if __name__ == '__main__':
    main()
