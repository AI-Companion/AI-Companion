import os
import re
import string
import tensorflow as tf
import marabou

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(marabou.__file__)))

MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")
PLOTS_DIR = os.path.join(ROOT_DIR, "saved_perf")
EMBEDDINGS_DIR = os.path.join(ROOT_DIR, "saved_embeddings")
DATA_DIR = os.path.join(ROOT_DIR, "saved_datasets")
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
NETWORK_DIR = os.path.join(ROOT_DIR, "saved_networks")

SA_CONFIG_FILE = os.path.join(ROOT_DIR, "config/config_sentiment_analysis.json")
TD_CONFIG_FILE = os.path.join(ROOT_DIR, "config/config_topic_classification.json")
NER_CONFIG_FILE = os.path.join(ROOT_DIR, "config/config_named_entity_recognition.json")
CC_CONFIG_FILE = os.path.join(ROOT_DIR, "config/config_fashion_classifier.json")


def custom_standardization(input_data):
    """
    Text tokenization function
    Args:
        input_data: raw input text
    Return:
        tokenized text
    """
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(
        stripped_html, '[%s]' % re.escape(string.punctuation), '')
