"""
http://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/
http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
http://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
"""


# import keras
from keras import callbacks
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense, Activation, Flatten
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

from os import listdir
from os.path import isfile, join

import numpy
import numpy as np


ds = '../dataset/training/wonderland.txt'
t_ds = '../dataset/training/wonderland.txt'

checkpoint = 'checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}-{acc:.4f}.hdf5'
load_checkpoint = ''


def char2vec(dataset):
    """Convert dataset into a one-hot encoded training data

    Total Vocab, is also the number of classes

    x: x_patterns, seq_length, features
    y: one hot encoding

    :param dataset:
    :return: x, y, n_chars, n_vocab, char_to_int, int_to_char
    """
    try:
        raw_text = open(dataset, 'r').read().lower()
        print('[*]', dataset)
    except:
        raise

    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    n_chars = raw_text.__len__()
    vocab_size = chars.__len__()
    features = vocab_size

    print('[*] Corpus Length:', n_chars)
    print('[*] Total Vocabulary (unique set):', vocab_size)

    X = []
    Y = []

    for i in range(0, n_chars - seq_length, 1):
        X.append(raw_text[i: i + seq_length])
        Y.append(raw_text[i + seq_length])

    n_patterns = X.__len__()
    samples = n_patterns
    timesteps = seq_length

    print('[*] Total Patterns', n_patterns)

    print('[*] Vectorization...')
    x = np.zeros((samples, timesteps, features), dtype=np.bool)
    y = np.zeros((samples, features), dtype=np.bool)
    for i, sentence in enumerate(X):
        for t, char in enumerate(sentence):
            x[i, t, char_to_int[char]] = 1
        y[i, char_to_int[Y[i]]] = 1


    # The input sequences that might vary in length between 1 and max_len and
    # therefore require zero padding. Here, we use left-hand-side
    # (prefix) padding with the Keras built in pad_sequences() function.
    # x = pad_sequences(x, maxlen=seq_length, dtype='float32')        # shape: (163717, 100)

    # We need to reshape the NumPy array into a format expected by
    # the LSTM networks, that is [samples, time step, features].
    # ;
    # The features in the input refers to attributes or columns in your data
    # ;
    # Each sample is a separate sequence. The sequence steps are time steps.
    # Each observation measurement is a feature.
    # ;
    # LSTM input must be 3D, the structure is [samples, timesteps, features].
    # If you one hot encode, then the number of labels (length of binary vectors)
    # is the number of features.
    # ;
    # (number of samples, number of timesteps, number of features)
    # x = numpy.reshape(x, (samples, timesteps, 1))

    # Once reshaped, we can then normalize the input integers to the
    # range 0-to-1, the range of the sigmoid activation functions used
    # by the LSTM network.
    # x = x / float(features)

    # Finally, we can think of this problem as a sequence classification
    # task, where each of the letters represents a different class. As
    # such, we can convert the output (y) to a one hot encoding, using the
    # Keras built-in function to_categorical().
    # y = np_utils.to_categorical(y)
    output = y.shape[1]

    return x, y, samples, timesteps, features, output, char_to_int, int_to_char


def word2vec(dataset):
    pass


def supervised(x, y):
    """
    The majority of practical machine learning uses supervised learning.

    - Classification: A classification problem is when the output variable
    is a category, such as “red” or “blue” or “disease” and “no disease”.

    - Regression: A regression problem is when the output variable is a real
    value, such as “dollars” or “weight”.

    :return:
    """
    pass


def unsupervised(x):
    """
    - Clustering: A clustering problem is where you want to discover
    the inherent groupings in the data, such as grouping customers by
    purchasing behavior.

    - Association:  An association rule learning problem is where you want
    to discover rules that describe large portions of your data, such as
    people that buy X also tend to buy Y.

    :param x:
    :return:
    """
    pass


def semisupervized(more_x, less_y):
    """
    Problems where you have a large amount of input data (X) and only
    some of the data is labeled (Y) are called semi-supervised learning
    problems.

    A good example is a photo archive where only some of the images are
    labeled, (e.g. dog, cat, person) and the majority are unlabeled.

    - You can use unsupervised learning techniques to discover and learn
    the structure in the input variables.

    - You can also use supervised learning techniques to make best guess
    predictions for the unlabeled data, feed that data back into the
    supervised learning algorithm as training data and use the model to
    make predictions on new unseen data.


    :param more_x:
    :param less_y:
    :return:
    """
    pass


def timesteps_dimension():
    """
    - timesteps: usually indicate the length of your sequence (each sample
    should have same length, or you should do some paddings of your input
    to have the same length)

    - dimension: usually indicate the number of units in a specific layer.
    for e.g. if your word embedding is 100 dimension, then you just set
    this param as 100.
    """
    pass


def time_dimension():
    """
    For a feed-forward network, your input has the shape
    (number of samples, number of features).

    With an LSTM/RNN, you add a time dimension, and your input shape becomes
    (number of samples, number of timesteps, number of features).
    This is in the documentation.

    So if your feature dimension is 5, and you have 2 timesteps, your input
    could look like:

    [ [[1,2,3,4,5], [2,3,4,5,6]], [[2,4,6,8,0], [9,8,7,6,5]] ]

    Your output shape depends on how you configure the net. If your LSTM/RNN
    has return_sequences=False, you'll have one label per sequence; if you set
    return_sequences=True, you'll have one label per timestep.
    """
    pass


def monitoring_fitting():
    """
    Overfitting: validation_loss > training_loss
    If your training loss is much lower than validation loss then this means the
    network might be overfitting. Solutions to this are to decrease your network
    size, or to increase dropout. For example you could try dropout of 0.5 and so on.

    Underfitting: training_loss == validation_loss
    If your training/validation loss are about equal then your model is underfitting.
    Increase the size of your model (either number of layers or the raw number of
    neurons per layer)
    """


def why_embedding():
    """
    https://www.quora.com/What-is-the-difference-between-using-word2vec-vs-one-hot-embeddings-as-input-to-classifiers

    1. One-hot vectors are high-dimensional and sparse, while word embeddings are
    low-dimensional and dense (they are usually between 50–600 dimensional). When
    you use one-hot vectors as a feature in a classifier, your feature vector grows
    with the vocabulary size; word embeddings are more computationally efficient.

    and more importantly:
    2. Word embeddings have the ability to generalize, due to semantically similar
    words having similar vectors, which is not the case in one-hot vectors (each pair
    of such vectors wi,wjwi,wj has cosine similarity cos(wi,wj)=0cos(wi,wj)=0).

    If your feature vector contains one-hot vectors of the documents’ words, you will
    only be able to consider features you’ve seen during training; when you use
    embeddings, semantically similar words will create similar features, and will lead
    to similar classification.

    """



# define the checkpoints
callbacks_list = [
    callbacks.ModelCheckpoint(checkpoint,
                              monitor='loss',
                              verbose=1,
                              save_best_only=True),
    callbacks.TerminateOnNaN()
]





#############################################################################################

batch_size = 128
seq_length = 40        # input_length
epochs = 1
initial_epoch = 0

x, y, samples, timesteps, features, output, char_to_int, int_to_char = char2vec(ds)
x_val, y_val = x, y

model = Sequential()

# LSTM: 3D, input_shape=(input_length, input_dim)
# Example: input_shape=(seq_length, features)
# Example: input_shape=(height, width)
# Example: reshape X to be [samples, time steps, features]
# Example: reshape X to be [samples, iterations, input vector]
# Example: (number_of_sequences, length_of_sequence, number_of_features)
# Documentation: (number of samples, number of timesteps, number of features)
# Example: model.add(LSTM(64, input_shape=(10,64)))

# Dense: 2D, input_shape=(batch_size, input_dim)

model.add(LSTM(128, input_shape=(timesteps, features)))  # (None, 100, 256)
print(model.output_shape)

model.add(Dropout(0.5))     # (None, 256)
print(model.output_shape)

# model.add(LSTM(128, return_sequences=True))    # (None, 100, 256)
# print(model.output_shape)
#
# model.add(Dropout(0.2))
# print(model.output_shape)

# model.add(Flatten())      # Flattens 3D -> 2D   # ValueError: Error when checking target: expected activation_1 to have 3 dimensions, but got array with shape (163717, 60)
# print(model.output_shape)

model.add(Dense(output))    # (None, 60)
print(model.output_shape)

model.add(Activation('relu'))   # (None, 60)
print(model.output_shape)





model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

#############################################################################################


if isfile(load_checkpoint):
    print('[*] Checkpoint loaded', load_checkpoint)
    model.load_weights(load_checkpoint)


# Updates happen after each batch
history = model.fit(x, y,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks_list,
                    validation_data=(x_val, y_val),
                    # validation_split=0.33
                    shuffle=False,
                    initial_epoch=initial_epoch)

loss, acc = history.history['loss'], history.history['acc']
print('loss:', loss, 'acc:', acc)

score, acc = model.evaluate(x, y, batch_size=batch_size, verbose=1)
print(score, acc)







