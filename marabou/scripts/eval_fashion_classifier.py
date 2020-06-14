import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import argparse
from typing import List
import numpy as np
from cv2 import cv2
from marabou.models.cnn_classifier import CNNClothing


def evaluate_model(image_url) -> None:
    """
    Wrapper function that calls the model deserializer and returns prediction
    Args:
        image: relative path for the model file
        questions_list: list of strings to perform inference on
    Return:
        list of probabilies for positive class for each input word
    """
    trained_model = CNNClothing.load_model()
    if trained_model is None:
        raise ValueError("there is no corresponding model file")
    im = cv2.imread(image_url)
    if im is not None:
            im = cv2.imread(image_url, 1)
            im = cv2.resize(im, (trained_model['image_width'], trained_model['image_height']))
    else:
        raise ValueError("please input a valid image url")
    image_class = trained_model.predict(np.asarray(im))
    print(image_class)


def parse_arguments():
    """
    Parse file arguments
    """
    parser = argparse.ArgumentParser(description="Predict class for a given input image")
    parser.add_argument('image_url', help="url for a test image")
    return parser.parse_args()


def main():
    """main function"""
    args = parse_arguments()
    evaluate_model(args.image_url)


if __name__ == '__main__':
    main()
