import os
import numpy as np
from itertools import compress
import re
import subprocess
import time
import gdown
from commons.definitions import ROOT_DIR, MODELS_DIR

def load_model(h5_file_url=None, class_file_url=None, preprocessor_file_url=None, collect_from_gdrive=False, use_case="ner"):
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
    if use_case == "ner":
        prefix = "named_entity_recognition"
    if use_case == "sa":
        prefix = "sentiment_analysis"
    if not collect_from_gdrive:
        model_files_list = os.listdir(MODELS_DIR)
        if len(model_files_list) > 0:
            rnn_models_idx = [(prefix in f) and ("rnn" in f) for f in model_files_list]
            if np.sum(rnn_models_idx) > 0:
                rnn_model = list(compress(model_files_list, rnn_models_idx))
                model_dates = [int(''.join(re.findall(r'\d+', f))) for f in rnn_model]
                h5_file_name = rnn_model[np.argmax(model_dates)]
                preprocessor_file = h5_file_name.replace("rnn_model.h5", "preprocessor.pkl")
                class_file = h5_file_name.replace("rnn_model.h5", "rnn_class.pkl")
                if (os.path.isfile(os.path.join(MODELS_DIR, preprocessor_file))) and\
                        (os.path.isfile(os.path.join(MODELS_DIR, class_file))):
                    h5_file_local_url = os.path.join(MODELS_DIR, h5_file_name)
                    class_file_local_url = os.path.join(MODELS_DIR, class_file)
                    preprocessor_file_local_url = os.path.join(MODELS_DIR, preprocessor_file)
                    return h5_file_local_url, class_file_local_url, preprocessor_file_local_url
                return None, None, None
            return None, None, None
        return None, None, None
    else:
        print("===========> collecting model file from link")
        file_prefix = "%s_loaded_%s" % (prefix, time.strftime("%Y%m%d_%H%M%S"))
        h5_file_name = file_prefix + "_rnn_model.h5"
        class_file_name = h5_file_name.replace("rnn_model.h5", "rnn_class.pkl")
        preprocessor_file_name = h5_file_name.replace("rnn_model.h5", "preprocessor.pkl")

        h5_file_local_url = os.path.join(MODELS_DIR, h5_file_name)
        class_file_local_url = os.path.join(MODELS_DIR, class_file_name)
        preprocessor_file_local_url = os.path.join(MODELS_DIR, preprocessor_file_name)

        h5_file_url = 'https://drive.google.com/uc?id={}'.format(h5_file_url)
        class_file_url = 'https://drive.google.com/uc?id={}'.format(class_file_url)
        preprocessor_file_url = 'https://drive.google.com/uc?id={}'.format(preprocessor_file_url)
    
        gdown.download(h5_file_url, h5_file_local_url, quiet=True)
        print("---> h5 model file download under %s" %h5_file_local_url)
        gdown.download(class_file_url, class_file_local_url, quiet=True)
        print("---> class file downloaded under %s" %class_file_local_url)
        gdown.download(preprocessor_file_url, preprocessor_file_local_url, quiet=True)
        print("---> preprocessor file downloaded under %s" %preprocessor_file_local_url)

        if (os.path.isfile(h5_file_local_url) and os.path.isfile(class_file_local_url)):
            return h5_file_local_url, class_file_local_url, preprocessor_file_local_url
        else:
            return None, None, None
