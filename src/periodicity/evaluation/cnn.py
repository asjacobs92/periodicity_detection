import numpy as np

from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, GlobalAveragePooling1D, Flatten

from sklearn.model_selection import train_test_split

from transforms import swt
from utils import factory, plotter


def decomposed_signal_test():
    """ Main method """
    print('########### Generating dataset ###########')
    X, y = factory.make_signal_dataset()

    X_decomposed = []
    for signal in X:
        hp_list, lp_list = swt(signal, 'haar', levels=6)
        coeffs = hp_list.values()
        X_decomposed.append(np.array(list(coeffs)))

    X_decomposed = np.array(X_decomposed)

    y_binary = to_categorical(y)

    print('########### Dataset generated ###########')
    X_train, X_test, y_train, y_test = train_test_split(X_decomposed, y_binary, test_size=0.30)

    print('X TRAIN', X_train.size, X_train[0].size, X_train[0])

    window_size, num_windows = 100, 7
    num_outputs = y_train[0].size

    print('############### Model init ###############')
    model = Sequential()
    model.add(Conv1D(filters=window_size, kernel_size=10, activation='relu', input_shape=(window_size, num_windows)))
    model.add(MaxPooling1D())
    model.add(Conv1D(filters=window_size, kernel_size=10, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(num_outputs, activation='softmax'))
    print(model.summary())

    print('############# Model compile #############')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print('############### Model fit ###############')
    model.fit(X_train, y_train, validation_split=0.2, epochs=10, verbose=1)

    model.predict(X_test[:4])


def full_signal_test():
    """ Main method """
    print('########### Generating dataset ###########')
    X, y = factory.make_signal_dataset()

    X = np.expand_dims(X, axis=2)  # reshape (569, 30) to (569, 30, 1)
    y_binary = to_categorical(y)

    print('########### Dataset generated ###########')
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.30)

    window_size, num_windows = 100, 1
    num_outputs = y_train[0].size

    print('############### Model init ###############')
    model = Sequential()
    model.add(Conv1D(filters=window_size, kernel_size=10, activation='relu', input_shape=(window_size, num_windows)))
    model.add(MaxPooling1D())
    model.add(Conv1D(filters=window_size, kernel_size=10, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(num_outputs, activation='softmax'))
    print(model.summary())

    print('############# Model compile #############')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print('############### Model fit ###############')
    model.fit(X_train, y_train, validation_split=0.2, epochs=10, verbose=1)

    model.predict(X_test[:4])


def main():
    """ Main method """
    decomposed_signal_test()


if __name__ == '__main__':
    main()
