import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import argparse
from typing import List
from mlp.named_entity_recognition import NERConfigReader, NERPreprocessor, NERRNN
from train_module.utils import load_model, ROOT_DIR, NER_CONFIG_FILE


def evaluate_model(questions_list: List[str], valid_config: NERConfigReader) -> None:
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
    trained_model = NERRNN(h5_file=h5_model_file, class_file=class_file)
    pre_processor = NERPreprocessor(preprocessor_file=preprocessor_file)
    questions_list_encoded, questions_list_tokenized, n_tokens, _ = pre_processor.preprocess(questions_list)
    labels_list = trained_model.predict(questions_list_encoded, pre_processor.labels_to_idx, n_tokens)
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
    if ROOT_DIR is None:
        raise ValueError("please make sure to setup the environment variable MARABOU_ROOT to point for the root of the project")
    valid_config = NERConfigReader(NER_CONFIG_FILE)
    evaluate_model(qlist, valid_config)


if __name__ == '__main__':
    main()
