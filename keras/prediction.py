"""
http://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/
http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
http://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
"""


from keras.models import Model, load_model
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from os.path import isfile, join
import numpy as np
import sys


ds = '../dataset/training/wonderland.txt'
t_ds = '../dataset/training/wonderland.txt'

modelName = 'Embedding +64E +128LSTM +128LSTM +activation +dropout'


def char2vec(dataset):
    """Convert dataset into an integer array for an Embedding layer

    x: Embedded array
    y: one hot encoding array

    :param dataset:
    :return: x, y, samples, timestep, features, char_to_int, int_to_char
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
    timestep = seq_length

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

    X = x
    # x = x / features

    return X, x, y, samples, timestep, features, char_to_int, int_to_char


#############################################################################################


batch_size = 100
seq_length = 100        # input_length
epochs = 1000000
initial_epoch = 0
normalize = True

X, x, y, samples, timestep, features, char_to_int, int_to_char = char2vec(ds)

load_checkpoint = 'checkpoints/weights-improvement-16-1.7885-0.4745.hdf5'
model = load_model(load_checkpoint)

#############################################################################################


print(model.summary())

start = np.random.randint(0, samples - 1)
pattern = list(X[start])
print('Pattern:', ''.join([int_to_char[value] for value in pattern]))
results = []
indexes = []
confidence = []
for i in range(100):
    x_predict = np.reshape(pattern, (1, 100))
    if normalize:
        x_predict = x_predict / features
    prediction = model.predict(x_predict)
    index = np.argmax(prediction)
    confidence.append(prediction[0][index])
    result = int_to_char[index]
    results.append(result)
    indexes.append(str(index))
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
    sys.stdout.write(result)
# print('results:', ''.join(results))
print('>> END')
print('indexes:', ''.join(indexes))
print('confidence: {:.2f}%'.format(int(sum(confidence)/len(confidence)*100)))
