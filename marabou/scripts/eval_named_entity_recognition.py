import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import argparse
from typing import List
from marabou.models.named_entity_recognition_rnn import RNNModel, DataPreprocessor


def evaluate_model(questions_list: List[str]) -> None:
    """
    Wrapper function that calls the model deserializer and returns prediction
    :param model_file_url: relative path for the model file
    :param questions_list: list of strings to perform inference on
    :return: list of probabilies for positive class for each input word
    """
    pre_processor = None
    trained_model = None
    trained_model, preprocessor_file = RNNModel.load_model()
    if trained_model is None:
        raise ValueError("there is no corresponding model file")
    pre_processor = DataPreprocessor.load_preprocessor(preprocessor_file)
    questions_list, n_tokens = DataPreprocessor.preprocess_data(questions_list, pre_processor)
    labels_list = trained_model.predict(questions_list, pre_processor["labels_to_idx"], n_tokens)
    print(labels_list)


def parse_arguments():
    """
    Parse file arguments
    """
    parser = argparse.ArgumentParser(description="Predict NER for a given input text")
    parser.add_argument('text', help="texts separated by space, each text must be surrounded with quotes", nargs='+')
    return parser.parse_args()


def main():
    """main function"""
    args = parse_arguments()
    qlist = args.text
    evaluate_model(qlist)


if __name__ == '__main__':
    main()
