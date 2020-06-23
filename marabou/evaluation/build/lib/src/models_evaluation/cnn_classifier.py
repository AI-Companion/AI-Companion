import os
import pickle
import re
import subprocess
import time
from itertools import compress
import numpy as np
from keras.models import load_model
from marabou.utils.config_loader import FashionClassifierConfigReader

class CNNClothing:
    """
    Handles the RNN model
    """
    def __init__(self, *args, **kwargs):
        self.use_pretrained_cnn = None
        self.pretrained_network_path = None
        self.pretrained_network_name = None
        self.pretrained_layer = None
        self.model = None
        self.n_labels = None
        self.idx_to_labels = None
        self.batch_size = None
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
            self.image_height = pickle.load(f)
            self.image_width = pickle.load(f)
            self.idx_to_labels = pickle.load(f)

    def predict(self, X_test):
        """
        Inference method
        Args:
            X_test: predictors array
        Return:
            numpy array containing the class for token character in the sentence
        """
        probs = self.model.predict(X_test)
        labels = np.argmax(probs, axis=1)
        labels = [self.idx_to_labels[i] for i in labels]
        return labels

    def predict_proba(self, X_test):
        """
        Inference method
        Args:
            X_test: array of predictors
        Return:
            numpy array containing the probabilities of a positive review for each list entry
        """
        probs = self.model.predict(X_test)
        return probs

    @staticmethod
    def load_model(h5_file_url=None, class_file_url=None, collect_from_gdrive=False):
        """
        Extracts a model saved using the save_model function
        Args:
            h5_file_url: gdrive link for the trained model
            class_file_url: gdrive link for the class file
            collect_from_gdrive: whether to collect the model file from google drive
        Return:
            model object and a tokenizer object
        """
        trained_model = None
        model_dir = os.path.join(os.getcwd(), "~/gitub/marabou_tmp/models")
        if not collect_from_gdrive:
            model_files_list = os.listdir(os.path.join(os.getcwd(), "~/gitub/marabou_tmp/models"))
            if len(model_files_list) > 0:
                rnn_models_idx = [("fashion_imagenet" in f) and ("rnn" in f) for f in model_files_list]
                if np.sum(rnn_models_idx) > 0:
                    rnn_model = list(compress(model_files_list, rnn_models_idx))
                    model_dates = [int(''.join(re.findall(r'\d+', f))) for f in rnn_model]
                    h5_file_name = rnn_model[np.argmax(model_dates)]
                    class_file = h5_file_name.replace("rnn_model.h5", "rnn_class.pkl")
                    if os.path.isfile(os.path.join(model_dir, class_file)):
                        trained_model = CNNClothing(h5_file=os.path.join(model_dir, h5_file_name),
                                                    class_file=os.path.join(model_dir, class_file))
                        return trained_model
                    return None
                return None
            return None
        else:
            print("===========> collecting model file from link")
            script_path = os.path.join(os.getcwd(), "bash_scripts/load_fashion_model_file.sh")
            file_prefix = "fashion_imagenet_loaded_%s" % time.strftime("%Y%m%d_%H%M%S")
            h5_file_name = file_prefix + "_rnn_model.h5"
            class_file_name = h5_file_name.replace("rnn_model.h5", "rnn_class.pkl")
            h5_file_local_url = os.path.join(model_dir, h5_file_name)
            class_file_local_url = os.path.join(model_dir, class_file_name)
            subprocess.call("%s %s %s %s %s" % (script_path, h5_file_url,
                                                h5_file_local_url, class_file_url, class_file_local_url), shell=True)
            if (os.path.isfile(h5_file_local_url) and os.path.isfile(class_file_local_url)):
                trained_model = CNNClothing(h5_file=h5_file_local_url, class_file=class_file_local_url)
                return trained_model
            else:
                return None
