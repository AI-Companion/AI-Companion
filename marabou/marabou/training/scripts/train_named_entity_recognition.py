from typing import List, Tuple
import time, os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from marabou.training.datasets import KaggleDataset
from marabou.commons import EMBEDDINGS_DIR, NER_CONFIG_FILE, MODELS_DIR, NERConfigReader
from marabou.commons import custom_standardization

def save_perf(model_name, history):
    fig, axs = plt.subplots(1,2)
    fig.tight_layout()
    acc = history['accuracy']
    val_acc = history['val_accuracy']
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


def train_model(config: NERConfigReader) -> None:
    """
    training function which prints classification summary as as result
    Args:
        config: Configuration object containing parsed .json file parameters
    Return:
        None
    """
    X, y = [], []
    if config.dataset_name == "kaggle_ner":
        dataset = KaggleDataset(config.dataset_url)
        X, y_tagged = dataset.get_set()
    tags = [item for sublist in y_tagged for item in sublist]
    unique_labels = list(set(tags))
    unique_labels.remove("O") # remove the generic tag
    tags2index = {t:i + 1 for i,t in enumerate(unique_labels)}
    tags2index["O"] = 0
    n_tags = len(tags2index)
    y = [[tags2index[w] for w in s] for s in y_tagged]
    if config.experimental_mode:
        ind = np.random.randint(0, len(X), 500)
        X = [X[i] for i in ind]
        y = [y[i] for i in ind]
    X = [" ".join(x) for x in X]
    labels = tf.keras.preprocessing.sequence.pad_sequences(maxlen=config.max_sequence_length, sequences=y, padding="post", value=tags2index["O"])
    d_size = len(X)
    valid_size = (int) (d_size * config.validation_split)
    dataset = tf.data.Dataset.from_tensor_slices((X, labels))
    print("dataset spec {}".format(dataset.element_spec))
    valid_ds = dataset.take(valid_size).shuffle(10000).batch(64).prefetch(tf.data.AUTOTUNE)
    train_ds = dataset.skip(valid_size).shuffle(10000).batch(64).prefetch(tf.data.AUTOTUNE)

    # build and compile model
    tokenizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
                #ngrams=3,
                max_tokens=config.vocab_size,
                output_sequence_length=config.max_sequence_length,
                output_mode='int')
    tokenizer.adapt(dataset.map(lambda text, label: text))
    """
    # visualize tokenizer transformation

    vocab = np.array(tokenizer.get_vocabulary())
    for example, label in dataset.take(1):
        encoded_example = tokenizer(example)[:3].numpy()
        for n in range(3):
            print("Original: ", example[n].numpy())
            print("Round-trip: ", " ".join(vocab[encoded_example[n]]))
            print()
    """

    input_layer = tf.keras.Input(shape=(1,), dtype=tf.string, name='input')
    x = tokenizer(input_layer)
    x = tf.keras.layers.Embedding(input_dim=len(tokenizer.get_vocabulary()),
                                 output_dim=config.embedding_dimension,
                                 input_length=config.max_sequence_length, trainable=True)(x)    
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=50, return_sequences=True, recurrent_dropout=0.2, dropout=0.2))(x)
    x_rnn = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=50, return_sequences=True, recurrent_dropout=0.2, dropout=0.2))(x)
    x = tf.keras.layers.add([x, x_rnn])  # residual connection to the first biLSTM
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_tags, activation='softmax'))(x)
    model = tf.keras.Model(inputs=input_layer, outputs=x)
    print(model.summary())
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    optimizer='adam', metrics=["accuracy"])
    history = model.fit(train_ds, validation_data=valid_ds, epochs=config.n_iter)
    save_perf(config.model_name, history.history)
    tf.keras.models.save_model(model, os.path.join(MODELS_DIR, config.model_name))

def main():
    """main function"""
    train_config = NERConfigReader(NER_CONFIG_FILE)
    train_model(train_config)


if __name__ == '__main__':
    main()
