"""
http://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/
http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
http://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
"""


# import keras
from keras import callbacks
from keras.models import Sequential
from keras.layers import Embedding
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


def char2vec(dataset):
    """Convert dataset into an integer array for an Embedding layer

    x: Embedded array
    y: one hot encoding array

    :param dataset:
    :return: x, y, samples, timesteps, features, char_to_int, int_to_char
    """

    try:
        raw_text = open(dataset, 'r').read().lower()
        print('[*]', dataset)
    except:
        raise

    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    nb_chars = raw_text.__len__()
    features = chars.__len__()
    timesteps = seq_length

    # cut the text in semi-redundant sequences of seq_length

    step = 3
    X = []
    Y = []
    for i in range(0, nb_chars - seq_length, step):
        X.append(raw_text[i: i + seq_length])
        Y.append(raw_text[i + seq_length])

    samples = X.__len__()

    print('[*] Corpus Length:', nb_chars)   # 163817
    print('[*] Features:', features)        # 61
    print('[*] Samples:', samples)          # 163761
    print('[*] Timestep:', seq_length)      # 56

    # https://github.com/minimaxir/char-embeddings/blob/master/text_generator_keras.py#L48
    # x = np.zeros((len(sentences), maxlen), dtype=np.int)
    # y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    # for i, sentence in enumerate(sentences):
    #     for t, char in enumerate(sentence):
    #         X[i, t] = char_indices[char]
    #     y[i, char_indices[next_chars[i]]] = 1

    print('[*] Vectorization...')
    x = np.zeros((samples, seq_length), dtype=np.int32)
    y = np.zeros((samples, features), dtype=np.bool)
    for i, sentence in enumerate(X):
        for t, char in enumerate(sentence):
            x[i, t] = char_to_int[char]
        y[i, char_to_int[Y[i]]] = 1

    return x, y, samples, timesteps, features, char_to_int, int_to_char


def char2vec_onehot(dataset):
    """Convert dataset into a one-hot encoded training data

    Total Vocab, is also the number of classes

    x: x_patterns, seq_length, features
    y: one hot encoding

    :param dataset:
    :return: x, y, samples, timesteps, features, output, char_to_int, int_to_char
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

    return x, y, samples, timesteps, features, char_to_int, int_to_char


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
    # https://github.com/fchollet/keras/issues/4838#issuecomment-326688772

    @naisanza a one-hot encoding followed by a dense layer is the same as a single
    embedding layer. Try both and you should get the same results with different
    runtime. Do the linear algebra if you need to convince yourself.

    The other big difference is lets say you have 256 categories. Each sample could
    be one unsigned short (1 byte) or 256 floats (4*256 bytes). Passing data back and
    forth to the CPU, the former should be much faster.

    I try to never generate one-hot encodings on the CPU then send them to the GPU
    because it feels like such a waste. However, it is a lot easier to understand,
    so might be better for examples.

    You can feed embeddings into an LSTM as well. That would be like having one-hot,
    then dense, then LSTM, so one more layer than the current examples have.


    left = Sequential()
    left.add(Embedding(input_dim=csize, output_dim=rnn_size, input_length=q_seq_size, mask_zero=True))
    left.add(LSTM(rnn_size))

    right = Sequential()
    right.add(Embedding(input_dim=csize, output_dim=rnn_size, input_length=t_seq_size, mask_zero=True))
    right.add(LSTM(rnn_size))

    merged = Sequential()
    merged.add(Merge([left, right], mode='cos'))
    merged.compile(optimizer='adam', loss='mse')



    # https://www.quora.com/What-is-the-difference-between-using-word2vec-vs-one-hot-embeddings-as-input-to-classifiers

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


def lstm_shape():
    """
    The shape of an LSTM is a 3D, comprised of (in order):
        1. number of samples
        2. number of timesteps
        3. number of features

        (number of samples, number of timesteps, number of features)

    In keras, the source says LSTM input_shape=(input_length, input_dim)
    In the numerous examples around the internet "input_length" and "input_dim" have been seen as:
        - input_shape=(seq_length, features)
        - input_shape=(height, width)
        - input_shape=(timesteps, features)
        - input_shape=(iterations, input vector)
        - input_shape=(length_of_sequence, number_of_features)

        I know what you're thinking. So many, right? Time to brush up on those SAT synonyms.
        They all mean the same thing, but when you jump into this and everyone's using their own
        terminology for everything, you'll just have to get familiar with all of them.

    LSTM: 3D, input_shape=(input_length, input_dim)
    Example: reshape X to be [samples, time steps, features]
    Example: reshape X to be [samples, iterations, input vector]
    Example: (number_of_sequences, length_of_sequence, number_of_features)
    Documentation: (number of samples, number of timesteps, number of features)
    Example: model.add(LSTM(64, input_shape=(10, 64)))
    """


def dense_layer():
    """
    A dense layer is 2D, comprised of:
        1. batch size
        2. input dimension

    Dense: 2D, input_shape=(batch_size, input_dim)
    """


def flatten_layer():
    """
    Flattens an input from 3D -> 3D
    """


# define the checkpoints
checkpoint = 'checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}-{acc:.4f}.hdf5'
callbacks_list = [
    callbacks.ModelCheckpoint(checkpoint,
                              monitor='loss',
                              verbose=1,
                              save_best_only=True),
    callbacks.TerminateOnNaN()
]





#############################################################################################

batch_size = 100
seq_length = 56        # input_length
epochs = 1000000
initial_epoch = 17

x, y, samples, timesteps, features, char_to_int, int_to_char = char2vec(ds)
# x, y, samples, timesteps, features, char_to_int, int_to_char = char2vec_onehot(ds)
x_val, y_val = x, y

model = Sequential()

model.add(Embedding(output_dim=64, input_dim=features))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(features))
model.add(Activation('relu'))

# one-hot
# model.add(LSTM(64, input_shape=(timesteps, features), return_sequences=False))
# model.add(Dropout(0.2))     # (Srivastava, 2013)
# model.add(Dense(features))
# model.add(Activation('relu'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

print(model.summary())


#############################################################################################


load_checkpoint = 'checkpoints/weights-improvement-17-2.9099-0.2565.hdf5'

if isfile(load_checkpoint):
    try:
        model.load_weights(load_checkpoint)
        print('[*] Checkpoint loaded', load_checkpoint)
    except:
        print('[*] Checkpoint load failed')
        initial_epoch = 0
else:
    print('[*] No model loaded')
    initial_epoch = 0


# Updates happen after each batch
history = model.fit(x, y,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks_list,
                    # validation_data=(x_val, y_val),
                    # validation_split=0.33,
                    shuffle=False,
                    initial_epoch=initial_epoch)

# loss, acc = history.history['loss'], history.history['acc']
# print('loss:', loss, 'acc:', acc)

# score, acc = model.evaluate(x, y, batch_size=batch_size, verbose=1)
# print('score:', score, 'accuracy:', acc)
