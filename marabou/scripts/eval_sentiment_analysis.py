import warnings
warnings.filterwarnings('ignore')
import argparse
from typing import List
from marabou.models.sentiment_analysis.tf_idf_models import DumbModel
from marabou.models.sentiment_analysis.rnn_models import RNNModel
from marabou.utils.data_utils import DataPreprocessor
from marabou.utils.config_loader import ConfigReader


def evaluate_model(questions_list: List[str], valid_config: ConfigReader) -> None:
    """
    Wrapper function that calls the model deserializer and returns prediction
    :param model_file_url: relative path for the model file
    :param questions_list: list of strings to perform inference on
    :return: list of probabilies for positive class for each input word
    """
    model = None
    pre_processor = None
    if valid_config.eval_model_name == "tfidf":
        model = DumbModel.load_model()
    if valid_config.eval_model_name == "rnn":
        model, preprocessor_file = RNNModel.load_model()
    if model is None:
        raise ValueError("there is no corresponding model file")
    if valid_config.eval_model_name == "rnn":
        pre_processor = DataPreprocessor.load_preprocessor(preprocessor_file)
        questions_list = DataPreprocessor.preprocess_data(questions_list, pre_processor)
    probs = model.predict_proba(questions_list)
    print(model.get_output(probs, questions_list))


def parse_arguments():
    """
    Parse file arguments
    """
    parser = argparse.ArgumentParser(description="Predict sentiment from a given text")
    parser.add_argument('--config', '-c', help='Path to the configuration file', required=True)
    parser.add_argument('question', help="text or list of texts to perform inference on")

    return parser.parse_args()


def main():
    """main function"""
    args = parse_arguments()
    qlist = args.question.strip('][').split(',')
    valid_config = ConfigReader(args.config)
    evaluate_model(qlist, valid_config)


if __name__ == '__main__':
    main()
