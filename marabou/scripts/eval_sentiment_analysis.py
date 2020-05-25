import warnings
warnings.filterwarnings('ignore')
import argparse
from typing import List
from marabou.models.sentiment_analysis.tf_idf_models import DumbModel
from marabou.models.sentiment_analysis.rnn_models import RNNModel, DataPreprocessor
from marabou.utils.config_loader import ConfigReader


def evaluate_model(questions_list: List[str], valid_config: ConfigReader) -> None:
    """
    Wrapper function that calls the model deserializer and returns prediction
    :param model_file_url: relative path for the model file
    :param questions_list: list of strings to perform inference on
    :return: list of probabilies for positive class for each input word
    """
    pre_processor = None
    trained_model = None
    if valid_config.eval_model_name == "tfidf":
        trained_model = DumbModel.load_model()
    if valid_config.eval_model_name == "rnn":
        trained_model, preprocessor_file = RNNModel.load_model()
    if trained_model is None:
        raise ValueError("there is no corresponding model file")
    if valid_config.eval_model_name == "rnn":
        pre_processor = DataPreprocessor.load_preprocessor(preprocessor_file)
        questions_list = DataPreprocessor.preprocess_data(questions_list, pre_processor)
    if valid_config.eval_model_name not in ["rnn", "tfidf"]:
        raise ValueError("please set eva_mode_name to be either 'rnn' or 'tfidf'")
    probs = trained_model.predict_proba(questions_list)
    print(probs)
    preds = trained_model.predict(questions_list)
    print(preds)


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
