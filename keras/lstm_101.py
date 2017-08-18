
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense, Activation



model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))



x_data = []
y_data = []

x_train = []
y_train = []

x_test = []
y_test = []


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

hidden_state = 256
timesteps = x_train.shape[1]
data_dim = x_train.shape[2]     # features?
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





