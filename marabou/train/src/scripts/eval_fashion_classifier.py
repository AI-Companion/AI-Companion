import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import argparse
import numpy as np
from cv2 import cv2
from src.models.cnn_classifier import CNNClothing
from src.utils.config_loader import FashionClassifierConfigReader


def evaluate_model(image_url: str, config: FashionClassifierConfigReader) -> None:
    """
    Wrapper function that calls the model deserializer and returns prediction
    Args:
        image: relative path for the model file
        questions_list: list of strings to perform inference on
    Return:
        list of probabilies for positive class for each input word
    """
    trained_model = CNNClothing.load_model(config.h5_model_url, config.class_file_url, collect_from_gdrive=False)
    if trained_model is None:
        raise ValueError("there is no corresponding model file")
    im = cv2.imread(image_url)
    if im is not None:
        im = cv2.imread(image_url, 1)
        im = cv2.resize(im, (trained_model.image_height, trained_model.image_width))
        im = np.asarray(im).reshape(1, trained_model.image_height, trained_model.image_width, 3)
    else:
        raise ValueError("please input a valid image url")
    image_class = trained_model.predict(im)
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
    root_dir = os.environ.get("MARABOU_HOME")
    if root_dir is None:
        raise ValueError("please make sure to setup the environment variable MARABOU_ROOT to\
                         point for the root of the project")
    config_file_path = os.path.join(root_dir, "marabou/train/config/config_fashion_classifier.json")
    config_file = FashionClassifierConfigReader(config_file_path)
    evaluate_model(args.image_url, config_file)


if __name__ == '__main__':
    main()
