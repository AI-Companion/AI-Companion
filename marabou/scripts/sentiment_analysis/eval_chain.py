import argparse
from typing import List
from marabou.models.dumb_model import DumbModel


def ask_model(model_file_url: str, questions_list: List[str]) -> None:
    """
    Wrapper function that calls the model deserializer and returns prediction
    :param model_file_url: relative path for the model file
    :param questions_list: list of strings to perform inference on
    :return: list of probabilies for positive class for each input word
    """
    print(f'Asking model {model_file_url} about "{questions_list}"')

    model = DumbModel.deserialize(model_file_url)
    probs = model.predict_proba(questions_list)
    print(model.get_output(probs, questions_list))


def parse_arguments():
    """
    Parse file arguments
    """
    parser = argparse.ArgumentParser(description="predict sentiment from a given text")
    parser.add_argument('model_file', help='model file', type=str)
    parser.add_argument('question', help="text to perform inference on")

    return parser.parse_args()


def main():
    """main function"""
    args = parse_arguments()
    qlist = args.question.strip('][').split(',')

    ask_model(args.model_file, qlist)


if __name__ == '__main__':
    main()
