import time
from typing import List, Tuple, Dict
import os
import re
import string
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from marabou.commons import ROOT_DIR, MODELS_DIR, SA_CONFIG_FILE, SAConfigReader, EMBEDDINGS_DIR


def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')

def save_perf(model_name, history):
    fig, axs = plt.subplots(1,2)
    fig.tight_layout()
    acc = history['binary_accuracy']
    val_acc = history['val_binary_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    axs[0].plot(epochs, loss, 'bo', label='Training loss')
    axs[0].plot(epochs, val_loss, 'b', label='Validation loss')
    axs[0].set_title('Training and validation loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(epochs, acc, 'bo', label='Training accuracy')
    axs[1].plot(epochs, val_acc, 'b', label='Validation accuracy')
    axs[1].set_title('Training and validation accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend(loc='lower right')

    output_file_name = os.path.join(MODELS_DIR, model_name)
    plt.savefig(output_file_name)

def train_model(config: SAConfigReader) -> None:
    """
    Training pipeline, input dataset is the sentiment analysis dataset hosted under 
    https://ai.stanford.edu/~amaas/data/sentiment/
    The pipeline will save a model object will be saved under saved_models along with model performances
    Args:
        config: Configuration object containing parsed .json file parameters
    Return:
        None
    """
    # load variables
    validation_split = (int)(100 - 100 * config.validation_split)
    vocab_size = config.vocab_size
    # max_sequence_length = config.max_sequence_length
    n_iter = config.n_iter
    buffer_size = 3000
    train_size = config.train_size
    batch_size = config.batch_size
    # load dataset
    train_ds, valid_ds, test_ds = tfds.load(name="imdb_reviews",
                                            split=('train[:{}%]'.format(validation_split), 'train[{}%:]'.format(validation_split), 'test'),
                                            as_supervised=True)
    if train_size < 1:
        n_train_total = train_ds.cardinality().numpy()
        n_train = (int)(train_size * n_train_total)
        train_ds = train_ds.take(n_train)
    train_ds = train_ds.cache().shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.cache().shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # build and compile model
    tokenizer_bow = tf.keras.layers.experimental.preprocessing.TextVectorization(
                        standardize=custom_standardization,
                        ngrams=3,
                        max_tokens=vocab_size,
                        output_mode='count')
    tokenizer_bow.adapt(train_ds.map(lambda text, label: text))
    inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
    x = tokenizer_bow(inputs)
    x = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0005, l2=0.0))(x)
    model = tf.keras.Model(inputs, x)
    print(model.summary())
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer='adam',
                metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
    
    # fit model
    history = model.fit(train_ds, validation_data=valid_ds, epochs=n_iter)
    save_perf(config.model_name, history.history)

    export_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Activation('sigmoid')
    ])
    export_model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
    )
    # save
    export_model.save(os.path.join(MODELS_DIR, config.model_name))


def main():
    """main function"""
    train_config = SAConfigReader(SA_CONFIG_FILE)
    train_model(train_config)


if __name__ == '__main__':
    main()
