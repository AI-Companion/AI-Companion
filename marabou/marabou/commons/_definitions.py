import os
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
