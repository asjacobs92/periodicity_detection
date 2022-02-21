import pandas
import math
import numpy as np
import matplotlib.pyplot as plt
import rootpath

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from keras import backend
from keras.layers import LSTM
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential

from periodicity_detection.utils import plotter, dataset, log
from periodicity_detection.transforms import swt


def r2(y_true, y_pred):
    # return r2_score(backend.get_value(y_true), backend.get_value(y_pred))
    SS_res = backend.sum(backend.square(y_true - y_pred))
    SS_tot = backend.sum(backend.square(y_true - backend.mean(y_true)))
    return (1 - SS_res / (SS_tot + backend.epsilon()))


def get_model(input_shape=(1, 1), output_shape=1):
    model = Sequential()
    model.add(LSTM(100, input_shape=input_shape))
    # model.add(LSTM(250))
    model.add(Dropout(0.2))
    # model.add(Dense(50))
    model.add(Dense(output_shape))
    model.add(Activation("linear"))
    return model


def raw_signal_test(dataset_path):
    """ Main method """
    logger = log.Logger("{}/res/log/raw_signal_test.log".format(rootpath.detect()))
    logger.log('########### RAW SIGNAL TEST START ###########'.format(dataset_path))
    logger.log('########### Reading dataset {} ###########'.format(dataset_path))
    X, y = dataset.read(dataset_path)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    logger.log('########### Dataset generated ###########')
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

    logger.log('X TRAIN Shape: {}'.format(X_train.shape))
    logger.log('Y TRAIN Shape: {}'.format(y_train.shape))

    points_per_step = X_train.shape[1]
    timesteps = int(X_train.shape[1] / points_per_step)

    X_train = np.reshape(X_train, (X_train.shape[0], timesteps, points_per_step))
    X_test = np.reshape(X_test, (X_test.shape[0], timesteps, points_per_step))

    logger.log('X TRAIN Shape: {}'.format(X_train.shape))
    logger.log('Y TRAIN Shape: {}'.format(y_train.shape))

    logger.log('############### Model init ###############')
    model = get_model(input_shape=(timesteps, points_per_step), output_shape=y_train.shape[1])
    logger.log(model.summary())

    callbacks = [log.LoggingCallback(logger.log)]

    logger.log('############# Model compile #############')
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', r2])

    logger.log('############### Model fit ###############')
    model.fit(X_train, y_train, validation_split=0.2, epochs=100,
              verbose=2, batch_size=1, workers=8, callbacks=callbacks)

    logger.log('############### Model test ###############')
    logger.log("Predict one: {}".format(model.predict(X_test[:10]), y_test[:10], callbacks=callbacks))
    y_pred = model.predict(X_test)
    logger.log('R2 Score for test data: {}'.format(r2_score(y_test, y_pred)))
    logger.log('########### RAW SIGNAL TEST END ###########'.format(dataset_path))


def decomposed_signal_test(dataset_path):
    """ Main method """
    logger = log.Logger("{}/res/log/decomposed_signal_test.log".format(rootpath.detect()))
    logger.log('########### DECOMPOSED SIGNAL TEST START ###########'.format(dataset_path))
    logger.log('########### Reading dataset {} ###########'.format(dataset_path))
    X, y = dataset.read(dataset_path)

    levels = 5
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_decomposed = []
    for signal in X:
        hp_list, lp_list = swt(signal, 'haar', levels=levels)
        coeffs = hp_list.values()
        scaled_coeffs = scaler.fit_transform(list(coeffs))
        X_decomposed.append(np.array(scaled_coeffs))
    X_decomposed = np.array(X_decomposed)

    logger.log('########### Dataset generated ###########')
    X_train, X_test, y_train, y_test = train_test_split(X_decomposed, y, test_size=0.3)

    logger.log('X TRAIN Shape: {}'.format(X_train.shape))
    logger.log('Y TRAIN Shape: {}'.format(y_train.shape))

    logger.log('############### Model init ###############')
    model = get_model(input_shape=(X_train.shape[1], X_train.shape[2]), output_shape=y_train.shape[1])
    logger.log(model.summary())

    callbacks = [log.LoggingCallback(logger.log)]

    logger.log('############# Model compile #############')
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', r2])

    logger.log('############### Model fit ###############')
    model.fit(X_train, y_train, validation_split=0.2, epochs=100, verbose=2,
              batch_size=1, workers=8, callbacks=callbacks)

    logger.log('############### Model test ###############')
    logger.log("Predict one: {}".format(model.predict(X_test[:10]), y_test[:10], callbacks=callbacks))
    y_pred = model.predict(X_test)
    logger.log('R2 Score for test data: {}'.format(r2_score(y_test, y_pred)))
    logger.log('########### DECOMPOSED SIGNAL TEST END ###########'.format(dataset_path))


def main():
    """ Main method """
    np.random.seed(7)

    # 1000 entries, 1000 points
    # raw_signal_test("{}/res/datasets/signals/single_clean_1000_1000.csv".format(rootpath.detect()))
    # raw_signal_test("{}/res/datasets/signals/single_noise_1000_1000.csv".format(rootpath.detect()))
    # raw_signal_test("{}/res/datasets/signals/multiple_clean_1000_1000.csv".format(rootpath.detect()))
    # raw_signal_test("{}/res/datasets/signals/multiple_noise_1000_1000.csv".format(rootpath.detect()))

    # decomposed_signal_test("{}/res/datasets/signals/single_clean_1000_1000.csv".format(rootpath.detect()))
    # decomposed_signal_test("{}/res/datasets/signals/single_noise_1000_1000.csv".format(rootpath.detect()))
    # decomposed_signal_test("{}/res/datasets/signals/multiple_clean_1000_1000.csv".format(rootpath.detect()))
    # decomposed_signal_test("{}/res/datasets/signals/multiple_noise_1000_1000.csv".format(rootpath.detect()))

    # 10000 entries, 1000 points
    raw_signal_test("{}/res/datasets/signals/single_clean_10000_1000.csv".format(rootpath.detect()))
    raw_signal_test("{}/res/datasets/signals/single_noise_10000_1000.csv".format(rootpath.detect()))
    raw_signal_test("{}/res/datasets/signals/multiple_clean_10000_1000.csv".format(rootpath.detect()))
    raw_signal_test("{}/res/datasets/signals/multiple_noise_10000_1000.csv".format(rootpath.detect()))

    decomposed_signal_test("{}/res/datasets/signals/single_clean_10000_1000.csv".format(rootpath.detect()))
    decomposed_signal_test("{}/res/datasets/signals/single_noise_10000_1000.csv".format(rootpath.detect()))
    decomposed_signal_test("{}/res/datasets/signals/multiple_clean_10000_1000.csv".format(rootpath.detect()))
    decomposed_signal_test("{}/res/datasets/signals/multiple_noise_10000_1000.csv".format(rootpath.detect()))


if __name__ == '__main__':
    main()
