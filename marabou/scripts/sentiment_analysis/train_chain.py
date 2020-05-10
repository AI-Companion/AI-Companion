import os
import argparse
from sklearn.metrics import classification_report
from marabou.utils.data_utils import ImdbDataset
from marabou.models.tf_idf_models import DumbModel


def train_model(model_type: str, vocab_size: int) -> None:
    """
    training function which prints classification summary as as result
    :param model_type: model name to be used
    :param vocab_size: number of rows to read from the training dataset
    :return: None
    """
    print(f'Training model from directory data/Imdb')

    dataset = ImdbDataset(vocab_size)
    X, y = dataset.get_set("train")
    train_data_size = len(X)
    print("Training data size: %i" % train_data_size)

    model = DumbModel(vocab_size=train_data_size)
    model.train(X, y)

    model_dir = os.path.join(os.getcwd(), "models/sentiment_analysis/")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    model_file_name = os.path.join(model_dir, model_type)

    print(f'Storing model to {model_file_name}')
    model.serialize(model_file_name)

    X_test, y_test = dataset.get_set("test")
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))


def parse_arguments():
    """script arguments parser"""
    parser = argparse.ArgumentParser(description="Train sentiment analysis classifier")
    parser.add_argument('model_type', help='model type', type=str)
    parser.add_argument('--vocab_size', help='volcabulary size')

    return parser.parse_args()


def main():
    """main function"""
    args = parse_arguments()
    if args.vocab_size is None:
        args.vocab_size = 0
    train_model(args.model_type, int(args.vocab_size))


if __name__ == '__main__':
    main()
