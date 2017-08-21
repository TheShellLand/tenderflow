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


ds = '../dataset/training/wonderland.txt'
t_ds = '../dataset/training/wonderland.txt'

checkpoint = 'checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}-{acc:.4f}.hdf5'
load_checkpoint = ''


def char2vec(dataset):
    """Convert dataset into training data

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

    print('[*] Total Characters:', n_chars)
    print('[*] Total Vocabulary (unique set):', vocab_size)

    x = []
    y = []

    for i in range(0, n_chars - seq_length, 1):
        sequence_in = raw_text[i:i + seq_length]    # 'project gutenberg’s alice’s adventures in wonderland, by lewis carroll this ebook is for the use o'
        sequence_out = raw_text[i + seq_length]     # 'f'

        x.append(list(map(lambda x: char_to_int[x], sequence_in)))
        y.append(char_to_int[sequence_out])

    n_patterns = x.__len__()
    time_steps = n_patterns
    samples = n_patterns

    print('[*] Total Patterns', n_patterns)

    # The input sequences that might vary in length between 1 and max_len and
    # therefore require zero padding. Here, we use left-hand-side
    # (prefix) padding with the Keras built in pad_sequences() function.
    x = pad_sequences(x, maxlen=seq_length, dtype='float32')

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
    x = numpy.reshape(x, (n_patterns, seq_length, features))

    # Once reshaped, we can then normalize the input integers to the
    # range 0-to-1, the range of the sigmoid activation functions used
    # by the LSTM network.
    x = x / float(vocab_size)

    # Finally, we can think of this problem as a sequence classification
    # task, where each of the letters represents a different class. As
    # such, we can convert the output (y) to a one hot encoding, using the
    # Keras built-in function to_categorical().
    y = np_utils.to_categorical(y)

    return x, y, n_chars, vocab_size, n_patterns, char_to_int, int_to_char


def word2vec(dataset):
    pass





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
seq_length = 100        # input_length
features = 1            # input_dim  # input vector shape(?)
epochs = 1000000
initial_epoch = 121

x_train, y_train, n_chars, vocab_size, n_patterns, char_to_int, int_to_char = char2vec(ds)
test = char2vec(t_ds)

model = Sequential()

# LSTM: 3D, input_shape=(input_length, input_dim)
# Example: input_shape=(seq_length, features)
# Example: input_shape=(height, width)
# Example: reshape X to be [samples, time steps, features]
# Example: reshape X to be [samples, iterations, input vector]
# Example: (number_of_sequences, length_of_sequence, number_of_features)
# model.add(LSTM(64, input_shape=(10,64)))

# Dense: 2D, input_shape=(batch_size, input_dim)

model.add(LSTM(vocab_size, input_shape=(seq_length, features), return_sequences=True))
print(model.output_shape)
model.add(Dropout(0.5))
print(model.output_shape)
model.add(LSTM(vocab_size, return_sequences=True))
print(model.output_shape)
model.add(Flatten())
print(model.output_shape)
model.add(Dense(y_train.shape[1]))
print(model.output_shape)
model.add(Activation('relu'))
print(model.output_shape)

# (None, 61)        <-- LSTM(vocab_size, input_shape=(seq_length, features))

# (None, 100, 61)
# (None, 100, 61)
# (None, 100, 61)
# (None, 100, 60)
# (None, 100, 60)

# (None, 100, 61)
# (None, 100, 61)
# (None, 100, 61)
# (None, 6100)      <-- Flatten
# (None, 60)
# (None, 60)


model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

#############################################################################################


if isfile(load_checkpoint):
    print('[*] Checkpoint loaded', load_checkpoint)
    model.load_weights(load_checkpoint)


# Updates happen after each batch
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=callbacks_list,
          validation_data=(test[0], test[1]),
          shuffle=True,
          initial_epoch=initial_epoch)


scores = model.evaluate(x_train, y_train, verbose=1)


# for pattern in dataX:
#     x = numpy.reshape(pattern, (1, len(pattern), 1))
#     x = x / float(len(alphabet))
#     prediction = model.predict(x, verbose=0)
#     index = numpy.argmax(prediction)
#     result = int_to_char[index]
#     seq_in = [int_to_char[value] for value in pattern]
#     print(seq_in, "->", result)


# for i in range(20):
# 	pattern_index = numpy.random.randint(len(dataX))
# 	pattern = dataX[pattern_index]
# 	x = pad_sequences([pattern], maxlen=max_len, dtype='float32')
# 	x = numpy.reshape(x, (1, max_len, 1))
# 	x = x / float(len(alphabet))
# 	prediction = model.predict(x, verbose=0)
# 	index = numpy.argmax(prediction)
# 	result = int_to_char[index]
# 	seq_in = [int_to_char[value] for value in pattern]
# 	print seq_in, "->", result




