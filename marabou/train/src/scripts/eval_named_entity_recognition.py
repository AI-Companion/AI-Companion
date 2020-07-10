import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import argparse
from typing import List
from src.models.named_entity_recognition_rnn import RNNModel, DataPreprocessor
from src.utils.config_loader import NamedEntityRecognitionConfigReader


def evaluate_model(questions_list: List[str], config: NamedEntityRecognitionConfigReader) -> None:
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
    trained_model, preprocessor_file = RNNModel.load_model(config.h5_model_url, config.class_file_url,
                                                           config.preprocessor_file_url, collect_from_gdrive=False)
    if trained_model is None:
        raise ValueError("there is no corresponding model file")
    pre_processor = DataPreprocessor.load_preprocessor(preprocessor_file)
    questions_list_encoded, questions_list_tokenized, n_tokens =\
        DataPreprocessor.preprocess_data(questions_list, pre_processor)
    labels_list = trained_model.predict(questions_list_encoded, pre_processor["labels_to_idx"], n_tokens)
    display_result = trained_model.visualize(questions_list_tokenized, labels_list)
    print(display_result)


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
    root_dir = os.environ.get("MARABOU_HOME")
    config_file_path = os.path.join(root_dir, "marabou/train/config/config_named_entity_recognition.json")
    config_file = NamedEntityRecognitionConfigReader(config_file_path)
    evaluate_model(qlist, config_file)


if __name__ == '__main__':
    main()
