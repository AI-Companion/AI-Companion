import os
import numpy as np
from itertools import compress
import re
import subprocess
import time

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
    root_dir = os.environ.get("MARABOU_HOME")
    if not os.path.isdir(os.path.join(root_dir, "marabou/evaluation/trained_models")):
        os.mkdir(os.path.join(root_dir, "marabou/evaluation/trained_models"))
    model_dir = os.path.join(root_dir, "marabou/evaluation/trained_models")
    if not collect_from_gdrive:
        model_files_list = os.listdir(model_dir)
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
                    h5_file_local_url = os.path.join(model_dir, h5_file_name)
                    class_file_local_url = os.path.join(model_dir, class_file)
                    preprocessor_file_local_url = os.path.join(model_dir, preprocessor_file)
                    return h5_file_local_url, class_file_local_url, preprocessor_file_local_url
                return None, None, None
            return None, None, None
        return None, None, None
    else:
        bash_script_folder = os.path.join(root_dir, "marabou/train/bash_scripts")
        print("===========> collecting model file from link")
        script_path = os.path.join(bash_script_folder, "load_model_files.sh")
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
            return h5_file_local_url, class_file_local_url, preprocessor_file_local_url
        else:
            return None, None, None
