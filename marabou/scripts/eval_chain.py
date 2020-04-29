import os
import argparse
from sklearn.metrics import classification_report

from marabou.dumb_model import DumbModel

def ask_model(model_file, question):
    print(f'Asking model {model_file} about "{question}"')

    model = DumbModel.deserialize(model_file)

    y_pred = model.predict_proba([question])
    print(y_pred[0])

def parse_arguments():
    parser = argparse.ArgumentParser(description="predict sentiment from a given text")
    parser.add_argument('model_file', help='model file', type=str)
    parser.add_argument('question', help="text to perform inference on")

    return parser.parse_args()


def main():
    args = parse_arguments()

    ask_model(args.model_file, args.question)

if __name__ == '__main__':
    main()