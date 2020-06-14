import os
import string
import pickle
import re
from itertools import compress
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cv2 import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.applications.vgg16 import VGG16
from keras.models import Model, load_model
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from marabou.utils.config_loader import FashionClassifierConfigReader


class DataPreprocessor:
    """
    Utility class performing several data preprocessing steps
    """
    def __init__(self, config:FashionClassifierConfigReader):
        self.validation_split = config.validation_split
        self.image_height = config.image_height
        self.image_width = config.image_width

    def split_train_test(self, X, y):
        """
        Wrapper method to split training data into a validation set and a training set
        Args:
            X: tokenized predictors
            y: labels
        Returns:
            tuple consisting of training predictors, training labels, validation predictors, validation labels
        """
        print("===========> data split")
        unique_labels = list(set(y))
        n_labels = len(unique_labels)
        labels_to_idx = {t: i for i, t in enumerate(unique_labels)}
        idx_to_labels = {i: t for i, t in enumerate(unique_labels)}
        y = [ labels_to_idx[i] for i in y]
        y = to_categorical(y, num_classes=n_labels)
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=self.validation_split)
        print("----> data split finish")
        print('training features shape ', X_train.shape)
        print('testing features shape ', X_test.shape)
        print('training target shape ', np.asarray(y_train).shape)
        print('testing target shape ', np.asarray(y_test).shape)
        return X_train, X_test, np.asarray(y_train), np.asarray(y_test), idx_to_labels

    def load_images(self, X):
        X_result = []
        for image_url in X:
            im = cv2.imread(image_url, 1)                
            im = cv2.resize(im, (self.image_width, self.image_height))
            X_result.append(im)
        X_result = np.asarray(X_result)
        return X_result


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
        keys = kwargs.keys()
        if 'config' in keys:
            self.init_from_config_file(args[0], kwargs['config'])
        else:
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
            self.use_pretrained_cnn = pickle.load(f)
            self.pretrained_network_path = pickle.load(f)

    def init_from_config_file(self, idx_to_labels, config: FashionClassifierConfigReader):
        """
        initialize the class for the first time from a given configuration file and data processor
        Args:
            idx_to_labels: conversion from indices to original labels
            config: .json configuration reader
        Return:
            None
        """
        self.use_pretrained_cnn = config.pre_trained_cnn
        self.pretrained_cnn_name = config.pretrained_network_name
        self.model = None
        self.n_iter = 10
        self.image_height = config.image_height
        self.image_width = config.image_width
        self.idx_to_labels = idx_to_labels
        self.n_labels = len(idx_to_labels)
        if self.pretrained_network_name == "vgg16":
            self.pretrained_network_path = config.pretrained_network_vgg
        elif self.pretrained_network_name == "lenet":
            self.pretrained_network_path = config.pretrained_network_lenet
        self.model = self.build_model()

    def build_model(self):
        """
        Builds an CNN model according to fixed architecture
        Return:
            None
        """
        print("===========> build model")
        vggmodel = VGG16(include_top=False, input_shape=(self.image_height, self.image_width, 3))
        for layer in vggmodel.layers:
	        layer.trainable = False
        x = vggmodel.layers[-1].output
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(self.n_labels, activation='softmax')(x)
        # define new model
        model = Model(inputs=vggmodel.inputs, outputs=x)
        # summarize
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['acc'])
        print(model.summary())
        return model

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        """
        Fits the model object to the data
        Args:
            X_train: numpy array containing encoded training features
            y_train: numpy array containing training targets
            X_test: numpy array containing encoded test features
            y_test: numpy array containing test targets
        Return:
            list of values related to each datasets and loss function
        """
        if (X_test is not None) and (y_test is not None):
            history = self.model.fit(x=X_train, y=y_train, epochs=self.n_iter,
                                     batch_size=128, validation_data=(X_test, y_test),
                                     verbose=2)
        else:
            history = self.model.fit(x=X_train, y=y_train, epochs=self.n_iter, batch_size=128, verbose=2)
        return history

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
        probs = self.model.predict(encoded_text_list)
        return [p[0] for p in probs]

    def save_model(self, file_name_prefix):
        """
        Saves the trained model into a h5 file
        Args:
            file_name_prefix: a file name prefix having the following format 'sentiment_analysis_%Y%m%d_%H%M%S'
        Return:
            None
        """
        model_folder = os.path.join(os.getcwd(), "models")
        if not os.path.isdir(model_folder):
            os.mkdir(model_folder)
        file_url_keras_model = os.path.join(model_folder, file_name_prefix + "_rnn_model.h5")
        self.model.save(file_url_keras_model)
        file_url_class = os.path.join(model_folder, file_name_prefix + "_rnn_class.pkl")
        with open(file_url_class, 'wb') as handle:
            pickle.dump(self.use_pretrained_embedding, handle)
            pickle.dump(self.vocab_size, handle)
            pickle.dump(self.embedding_dimension, handle)
            pickle.dump(self.embeddings_path, handle)
            pickle.dump(self.max_length, handle)
            pickle.dump(self.word_index, handle)
        print("----> model saved to %s" % file_url_keras_model)
        print("----> class saved to %s" % file_url_class)

    def save_learning_curve(self, history, file_name_prefix):
        """
        Saves the learning curve plot
        Args:
            history: a dictionary object containing training and validation dataset loss function values and
            objective function values for each training iteration
            file_name_prefix: a file name prefix having the following format 'sentiment_analysis_%Y%m%d_%H%M%S'
        Return:
            None
        """
        plot_folder = os.path.join(os.getcwd(), "perf")
        if not os.path.isdir(plot_folder):
            os.mkdir(plot_folder)

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))

        fig, ax = plt.subplots(1, 2)
        ax[0].plot(epochs, acc, 'bo', label='Training acc')
        ax[0].plot(epochs, val_acc, 'b', label='Validation acc')
        ax[0].set_title('Training and validation accuracy')
        ax[0].legend()
        fig.suptitle('model performance')
        ax[1].plot(epochs, loss, 'bo', label='Training loss')
        ax[1].plot(epochs, val_loss, 'b', label='Validation loss')
        ax[1].set_title('Training and validation loss')
        ax[1].legend()
        plot_file_url = os.path.join(plot_folder, file_name_prefix + "_learning_curve.png")
        plt.savefig(plot_file_url)
        plt.close()
        print("----> learning curve saved to %s" % plot_file_url)

    @staticmethod
    def load_model():
        """
        Extracts a model saved using the save_model function
        Return:
            model object and a tokenizer object
        """
        trained_model = None
        model_dir = os.path.join(os.getcwd(), "models")
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
