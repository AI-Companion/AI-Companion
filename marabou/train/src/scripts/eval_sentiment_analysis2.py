import warnings
warnings.filterwarnings('ignore')
import argparse
import os
from typing import List
from mlp.sentiment_analysis import SAConfigReader, SAPreprocessor, SARNN
#from src.utils.config_loader import SentimentAnalysisConfigReader
from src.utils.load_models import load_model


def evaluate_model(questions_list: List[str], valid_config: SAConfigReader) -> None:
    """
    Wrapper function that calls the model deserializer and returns prediction
    Args:
        model_file_url: relative path for the model file
        questions_list: list of strings to perform inference on
    Return:
        list of probabilies for positive class for each input word
    """
    h5_model_file, class_file, preprocessor_file = load_model(h5_file_url=valid_config.h5_model_url,
                                                              class_file_url=valid_config.class_file_url,
                                                              preprocessor_file_url=valid_config.preprocessor_file_url,
                                                              collect_from_gdrive=False)
    if h5_model_file is None or preprocessor_file is None:
        raise ValueError("there is no corresponding model file")
    trained_model = SARNN(h5_file=h5_model_file, class_file=class_file)
    pre_processor = SAPreprocessor(preprocessor_file=preprocessor_file)
    questions_list = pre_processor.preprocess(questions_list)
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
    root_dir = os.environ.get("MARABOU_HOME")
    if root_dir is None:
        raise ValueError("please make sure to setup the environment variable MARABOU_ROOT to point for the root of the project")
    config_file_path = os.path.join(root_dir, "marabou/train/config/config_sentiment_analysis.json")
    valid_config = SAConfigReader(config_file_path)
    evaluate_model(qlist, valid_config)


if __name__ == '__main__':
    main()
