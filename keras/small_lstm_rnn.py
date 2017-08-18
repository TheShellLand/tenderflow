"""
http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
"""

import os
import numpy
import string
from keras import metrics
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras import backend as K



# load ascii text and convert to lowercase
filename = '../dataset/training/wonderland.txt'
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()


# create mapping of unique chars to integers
chars = sorted(list(set(raw_text))) # ['\n', ' ', '!', '#', '$', '%', '(', ')', '*', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', '@', '[', ']', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '‘', '’', '“', '”', '\ufeff']
char_to_int = dict((c, i) for i, c in enumerate(chars)) # {'a': 30, '1': 14, '-': 10, 'p': 45, '!': 2, ';': 24, 'z': 55, '(': 6, '_': 29, 'k': 40, 'j': 39, '7': 20, '’': 57, '#': 3, ' ': 1, '‘': 56, '4': 17, 'x': 53, 't': 49, 'g': 36, '”': 59, 'q': 46, 'u': 50, 'i': 38, 'b': 31, '?': 25, '\ufeff': 60, 'r': 47, 'v': 51, 'n': 43, ')': 7, '9': 22, 'c': 32, 's': 48, 'm': 42, '8': 21, '2': 15, '6': 19, '5': 18, 'l': 41, '@': 26, '0': 13, '[': 27, 'e': 34, '$': 4, '%': 5, '/': 12, 'y': 54, '.': 11, 'o': 44, '*': 8, 'w': 52, ':': 23, '“': 58, ',': 9, '\n': 0, 'd': 33, 'f': 35, 'h': 37, ']': 28, '3': 16}
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
print('[*] Total Characters:', n_chars) # Total Characters: 163817
print('[*] Total Vocab:', n_vocab)  # Total Vocab: 61

# prepare the dataset of input to output pairs encoded as integers (is this vectorizing?)
seq_length = 100
x_data = []  # <class 'list'>: [60, 45, 47, 44, 39, 34, 32, 49, 1, 36, 50, 49, 34, 43, 31, 34, 47, 36, 57, 48, 1, 30, 41, 38, 32, 34, 57, 48, 1, 30, 33, 51, 34, 43, 49, 50, 47, 34, 48, 1, 38, 43, 1, 52, 44, 43, 33, 34, 47, 41, 30, 43, 33, 9, 1, 31, 54, 1, 41, 34, 52, 38, 48, 1, 32, 30, 47, 47, 44, 41, 41, 0, 0, 49, 37, 38, 48, 1, 34, 31, 44, 44, 40, 1, 38, 48, 1, 35, 44, 47, 1, 49, 37, 34, 1, 50, 48, 34, 1, 44, 35, 1, 30, 43, 54, 44, 43, 34, 1, 30, 43, 54, 52, 37, 34, 47, 34, 1, 30, 49, 1, 43, 44, 1, 32, 44, 48, 49]
y_data = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    # x_data.append(list(map(lambda x: char_to_int[x], list(seq_in))))       #([char_to_int[char] for char in seq_in])   # list(lambda x: x[char], list(seq_in))
    x_data.append([char_to_int[char] for char in seq_in])   # list(lambda x: x[char], list(seq_in))
    y_data.append(char_to_int[seq_out])
n_patterns = len(x_data)
print('[*] Total Patterns:', n_patterns)    # Total Patterns: 163717

# reshape X to be [samples, time steps, features]
# (number_of_sequences, length_of_sequence, number_of_features)
x_train = numpy.reshape(x_data, (n_patterns, seq_length, 1))
# normalize
x_train = x_train / float(n_vocab)
# one hot encode the output variable
y_train = np_utils.to_categorical(y_data)





#############################################################################################
'''
# define the model
model = Sequential()
model.add(LSTM(256, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1], activation='relu'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
'''

hidden_state = 256              # output
timesteps = x_train.shape[1]    # height    # sequence length
data_dim = x_train.shape[2]     # width     # features
batch_size = 128
epochs = 1000000
initial_epoch = 0

def new_model():
    # define the model
    model = Sequential()
    model.add(LSTM(256,
                   input_shape=(x_train.shape[1], x_train.shape[2])))
    # model.add(LSTM(256,
    #                batch_size=batch_size,
    #                input_dim=timesteps,
    #                input_length=data_dim))
    model.add(Dropout(0.2))
    model.add(Dense(y_train.shape[1]))
    model.add(Activation(activation='relu'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=callbacks_list)

#############################################################################################









# define the checkpoint
filepath = 'checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}-{acc:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath,
                             monitor='loss',
                             verbose=1,
                             save_best_only=True)

callbacks_list = [checkpoint]


# load and resume from latest model
def newest_model(path):
    """Return the newest model in path"""
    ml = []

    if os.path.isdir(path):
        for item in os.listdir(path):
            item = path + '/' + item
            if os.path.exists(path):
                if not os.path.isdir(item):
                    ml.append(item)

    if not ml:
        return False

    ml = list(map(lambda file: file, ml))
    ml_stats = list(map(lambda x: x.st_mtime, [os.stat(file) for file in ml]))
    ml = dict(zip(ml_stats, ml))
    ml = ml[max(sorted(list(ml)))]

    return ml

new_m = newest_model('checkpoints')
if new_m:
    try:
        print('[*] Loading existing model', new_m)
        model = load_model(new_m)
        model.fit(x_train, y_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  callbacks=callbacks_list,
                  initial_epoch=initial_epoch)

    except ValueError as err:
        print('[*]', err)
        print('[*] Unable to import weights, creating new model')
        new_model()

else:
    new_model()




