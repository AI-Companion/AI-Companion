import os
import string
import pickle
import re
import time
import subprocess
from itertools import compress
from typing import List
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


class DataPreprocessor:
    """
    Utility class performing several data preprocessing steps
    """
    def __init__(self, max_sequence_length: int, validation_split: float, vocab_size: int):
        self.max_sequence_length = max_sequence_length
        self.validation_split = validation_split
        self.vocab_size = vocab_size
        self.tokenizer_obj = None

    @staticmethod
    def load_preprocessor(preprocessor_file_name):
        """
        Loads preprocessing tools for the model
        Args:
            preprocessor_file_name: data to evaluate
        Return:
            preprocessed object
        """
        preprocessor = {}
        preprocessor_file_name = os.path.join(os.getcwd(), "models", preprocessor_file_name)
        with open(preprocessor_file_name, 'rb') as f:
            preprocessor['tokenizer_obj'] = pickle.load(f)
            preprocessor['max_sequence_length'] = pickle.load(f)
        return preprocessor

    @staticmethod
    def preprocess_data(data, preprocessor):
        """
        Performs data preprocessing before inference
        Args:
            data: data to evaluate
            preprocessor: tokenizer object
        Return:
            preprocessed data
        """
        data = preprocessor['tokenizer_obj'].texts_to_sequences(data)
        data = pad_sequences(data, maxlen=preprocessor['max_sequence_length'], padding="post",
                             value=preprocessor['tokenizer_obj'].word_index["pad"])
        return data


class RNNModel:
    """
    Handles the RNN model
    """
    def __init__(self, **kwargs):
        self.model_name = "rnn"
        self.use_pretrained_embedding = None
        self.vocab_size = None
        self.embedding_dimension = None
        self.embeddings_path = None
        self.max_length = None
        self.word_index = None
        self.embedding_layer = None
        self.model = None
        keys = kwargs.keys()
        self.init_from_files(kwargs['h5_file'], kwargs['class_file'])

    def init_from_files(self, h5_file, class_file):
        """
        Initializes the class from a previously saved model
        Args:
            h5_file: url to a saved class
        Return:
            None
        """
        self.model = load_model(h5_file)
        with open(class_file, 'rb') as f:
            self.use_pretrained_embedding = pickle.load(f)
            self.vocab_size = pickle.load(f)
            self.embedding_dimension = pickle.load(f)
            self.embeddings_path = pickle.load(f)
            self.max_length = pickle.load(f)
            self.word_index = pickle.load(f)
        f.close()

    def predict(self, encoded_text_list):
        """
        Inference method
        Args:
            encoded_text_list: a list of texts to be evaluated. the input is assumed to have been
            preprocessed
        Return:
            numpy array containing the probabilities of a positive review for each list entry
        """
        probs = self.model.predict(encoded_text_list)
        boolean_result = probs > 0.5
        return [int(b) for b in boolean_result]

    def predict_proba(self, encoded_text_list):
        """
        Inference method
        Args:
            encoded_text_list: a list of texts to be evaluated. the input is assumed to have been
            preprocessed
        Return:
            numpy array containing the probabilities of a positive review for each list entry
        """
        print("bbbbbbbbbbbbbbbbbbbbbbbbbbb"+str(encoded_text_list))
        probs = self.model.predict(encoded_text_list)
        print("cccccccccccccccccccccccccc"+str(probs))
        return [p[0] for p in probs]

    @staticmethod
    def load_model(h5_file_url=None, class_file_url=None, preprocessor_file_url=None, collect_from_gdrive=False):
        """
        Extracts a model saved using the save_model function
        Args:
            h5_file_url: gdrive link for the trained model
            class_file_url: gdrive link for the class file
            preprocessor_file_url: gdrive link for the preprocessor file
            collect_from_gdrive: whether to collect the model file from google drive
        Return:
            model object and a tokenizer object
        """
        trained_model = None
        model_dir = os.path.join(os.getcwd(), "models")
        if not collect_from_gdrive:
            model_files_list = os.listdir(os.path.join(os.getcwd(), "models"))
            if len(model_files_list) > 0:
                rnn_models_idx = [("sentiment_analysis" in f) and ("rnn" in f) for f in model_files_list]
                if np.sum(rnn_models_idx) > 0:
                    rnn_model = list(compress(model_files_list, rnn_models_idx))
                    model_dates = [int(''.join(re.findall(r'\d+', f))) for f in rnn_model]
                    h5_file_name = rnn_model[np.argmax(model_dates)]
                    preprocessor_file = h5_file_name.replace("rnn_model.h5", "preprocessor.pkl")
                    class_file = h5_file_name.replace("rnn_model.h5", "rnn_class.pkl")
                    if (os.path.isfile(os.path.join(model_dir, preprocessor_file))) and\
                            (os.path.isfile(os.path.join(model_dir, class_file))):
                        trained_model = RNNModel(h5_file=os.path.join(model_dir, h5_file_name),
                                                 class_file=os.path.join(model_dir, class_file))
                        return trained_model, preprocessor_file
                    return None, None
                return None, None
            return None, None
        else:
            print("===========> collecting model file from link")
            script_path = os.path.join(os.getcwd(), "bash_scripts/load_named_entity_recognition_model_file.sh")
            file_prefix = "sentiment_analysis_loaded_%s" % time.strftime("%Y%m%d_%H%M%S")
            h5_file_name = file_prefix + "_rnn_model.h5"
            class_file_name = h5_file_name.replace("rnn_model.h5", "rnn_class.pkl")
            preprocessor_file_name = h5_file_name.replace("rnn_model.h5", "preprocessor.pkl")
            h5_file_local_url = os.path.join(model_dir, h5_file_name)
            class_file_local_url = os.path.join(model_dir, class_file_name)
            preprocessor_file_local_url = os.path.join(model_dir, preprocessor_file_name)
            subprocess.call("%s %s %s %s %s %s %s" % (script_path, h5_file_url, h5_file_local_url, class_file_url,
                                                      class_file_local_url, preprocessor_file_url,
                                                      preprocessor_file_local_url),
                            shell=True)
            if (os.path.isfile(h5_file_local_url) and os.path.isfile(class_file_local_url)):
                trained_model = RNNModel(h5_file=h5_file_local_url, class_file=class_file_local_url)
                return trained_model, preprocessor_file_name
            else:
                return None, None
