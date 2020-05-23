import argparse
from typing import List
from marabou.models.sentiment_analysis.tf_idf_models import DumbModel


def evaluate_model(questions_list: List[str]) -> None:
    """
    Wrapper function that calls the model deserializer and returns prediction
    :param model_file_url: relative path for the model file
    :param questions_list: list of strings to perform inference on
    :return: list of probabilies for positive class for each input word
    """
    model_file_url = "model"
    model = DumbModel.load_model(model_file_url)
    probs = model.predict_proba(questions_list)
    print(model.get_output(probs, questions_list))


def parse_arguments():
    """
    Parse file arguments
    """
    parser = argparse.ArgumentParser(description="Predict sentiment from a given text")
    parser.add_argument('question', help="text or list of texts to perform inference on")

    return parser.parse_args()


def main():
    """main function"""
    args = parse_arguments()
    qlist = args.question.strip('][').split(',')
    evaluate_model(qlist)


if __name__ == '__main__':
    main()
