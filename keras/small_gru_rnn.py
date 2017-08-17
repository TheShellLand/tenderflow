"""
http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
"""

import numpy
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
import os


# load ascii text and convert to lowercase
filename = 'dataset/training/wonderland.txt'
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text))) # ['\n', ' ', '!', '#', '$', '%', '(', ')', '*', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', '@', '[', ']', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '‘', '’', '“', '”', '\ufeff']
char_to_int = dict((c, i) for i, c in enumerate(chars)) # {'a': 30, '1': 14, '-': 10, 'p': 45, '!': 2, ';': 24, 'z': 55, '(': 6, '_': 29, 'k': 40, 'j': 39, '7': 20, '’': 57, '#': 3, ' ': 1, '‘': 56, '4': 17, 'x': 53, 't': 49, 'g': 36, '”': 59, 'q': 46, 'u': 50, 'i': 38, 'b': 31, '?': 25, '\ufeff': 60, 'r': 47, 'v': 51, 'n': 43, ')': 7, '9': 22, 'c': 32, 's': 48, 'm': 42, '8': 21, '2': 15, '6': 19, '5': 18, 'l': 41, '@': 26, '0': 13, '[': 27, 'e': 34, '$': 4, '%': 5, '/': 12, 'y': 54, '.': 11, 'o': 44, '*': 8, 'w': 52, ':': 23, '“': 58, ',': 9, '\n': 0, 'd': 33, 'f': 35, 'h': 37, ']': 28, '3': 16}

n_chars = len(raw_text)
n_vocab = len(chars)
print('Total Characters:', n_chars) # Total Characters: 163817
print('Total Vocab:', n_vocab)  # Total Vocab: 61

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print('Total Patterns:', n_patterns)    # Total Patterns: 163717

# reshape X to be [samples, time steps, features]
x_train = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
x_train = x_train / float(n_vocab)
# one hot encode the output variable
y_train = np_utils.to_categorical(dataY)
#############################################################################################

# define the model
model = Sequential()
# model.add(LSTM(256, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(GRU(256, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1], activation='relu'))
# model.add(Dropout(0.5))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#############################################################################################

# define the checkpoint
filepath = 'checkpoints/weights-improvement-GRU-relu-{epoch:02d}-{loss:.4f}-{acc:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath,
                             monitor='loss',
                             verbose=1,
                             save_best_only=True)

callbacks_list = [checkpoint]

# load and resume from latest model
newest_model = lambda files: os.listdir(files) if os.path.isdir(files) else None
newest_model = newest_model('checkpoints')

if newest_model:
    newest_model = list(map(lambda file: 'checkpoints' + '/' + file, newest_model))
    nm_stats = list(map(lambda x: x.st_mtime, [os.stat(file) for file in newest_model]))
    newest_model = dict(zip(nm_stats, newest_model))
    newest_model = newest_model[max(sorted(list(newest_model)))]
    print('Loading existing model', newest_model)

    model = load_model(newest_model)
    history = model.fit(x_train, y_train,
                        epochs=1000000,
                        batch_size=128,
                        callbacks=callbacks_list)
else:
    history = model.fit(x_train, y_train,
                        epochs=1000000,
                        batch_size=128,
                        callbacks=callbacks_list)



