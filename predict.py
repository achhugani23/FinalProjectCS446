import numpy as np
import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv3D, MaxPooling3D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K

model = Sequential()

def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend

def print_files(filename):
    data = np.load(filename)
    print(data.shape)

def create_convolution(train_X, train_Y):
    train_X = train_X.reshape(train_X.shape[0], 26, 31, 23, 1)
    train_X = train_X.astype('float32')
    model.add(Conv3D(32, (5, 5, 5), activation='relu', input_shape=(26,31,23,1)))
    model.add(Conv3D(32, (5, 5, 5), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2,2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(19, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    model.fit(train_X, train_Y,
          batch_size=32, epochs=32, verbose=1,
          callbacks = [ModelCheckpoint('model.h5', save_best_only=True)])

    # score = model.evaluate(train_X[:50], train_Y[:50], verbose=0)
    # print(score)

def predict(test_data):
    test_data = test_data.reshape(test_data.shape[0], 26, 31, 23, 1)
    test_data = test_data.astype('float32')
    preds = model.predict(test_data)
    preds[preds>=0.5] = 1
    preds[preds<0.5] = 0
    np.save('predictions.npy', preds)


if __name__ == "__main__":
    train_X = np.load("train_X.npy")
    train_Y = np.load("train_binary_Y.npy")
    set_keras_backend("theano")
    create_convolution(train_X, train_Y)
    test_x = np.load("valid_test_X.npy")
    predict(test_x)
