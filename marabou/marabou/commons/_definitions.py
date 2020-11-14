import os

ROOT_DIR = os.environ.get("MARABOU_HOME")
if ROOT_DIR is None:
    raise ValueError("please make sure to setup the environment variable MARABOU_ROOT to point for the root of the project")
MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")
PLOTS_DIR = os.path.join(ROOT_DIR, "saved_perf")
EMBEDDINGS_DIR = os.path.join(ROOT_DIR, "saved_embeddings")
DATA_DIR = os.path.join(ROOT_DIR, "saved_datasets")
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
NETWORK_DIR = os.path.join(ROOT_DIR, "saved_networks")

SA_CONFIG_FILE = os.path.join(ROOT_DIR, "config/config_sentiment_analysis.json")
NER_CONFIG_FILE = os.path.join(ROOT_DIR, "config/config_named_entity_recognition.json")
CC_CONFIG_FILE = os.path.join(ROOT_DIR, "config/config_fashion_classifier.json")
