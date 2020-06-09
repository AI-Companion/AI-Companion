import warnings
warnings.filterwarnings('ignore')
import argparse
from typing import List
from marabou.models.sentiment_analysis_tfidf import DumbModel
from marabou.models.sentiment_analysis_rnn import RNNModel, DataPreprocessor
from marabou.utils.config_loader import SentimentAnalysisConfigReader


def evaluate_model(questions_list: List[str], valid_config: SentimentAnalysisConfigReader) -> None:
    """
    Wrapper function that calls the model deserializer and returns prediction
    Args:
        model_file_url: relative path for the model file
        questions_list: list of strings to perform inference on
    Return:
        list of probabilies for positive class for each input word
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
    print("===========> probabilities")
    print(probs)
    # preds = trained_model.predict(questions_list)


def parse_arguments():
    """
    Parse file arguments
    """
    parser = argparse.ArgumentParser(description="Predict sentiment from a given text")
    parser.add_argument('question', help="text or list of texts to perform inference on", nargs='+')

    return parser.parse_args()


def main():
    """main function"""
    args = parse_arguments()
    qlist = args.question
    valid_config = SentimentAnalysisConfigReader("config/config_sentiment_analysis.json")
    evaluate_model(qlist, valid_config)


if __name__ == '__main__':
    main()
