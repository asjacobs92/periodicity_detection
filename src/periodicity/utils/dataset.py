""" Module to generate and manage signal and wavelets datasets """

import csv
import random
import numpy as np
import rootpath

from ast import literal_eval

# import factory
from . import factory


def write(path, dataset_size=1000, signal_size=1000, single=True, noise=True):
    """
        Generates datasets of random signals with random periodicities and writes in a CSV

        Args:
            - path
                Path to write generated dataset to
            - dataset_size
                Number of signals to generate
            - signal_size
                Number of points in the generated signals
            - single
                Wether generated signals should have single of multiple periodicities
            - noise
                Wether generated signals should have noise or not
        Returns: None
    """

    print("---- Generating dataset with (dataset_size={}, signal_size={}, single={}, noise={}) ----".format(dataset_size, signal_size, single, noise))
    X = []
    y = []
    for idx in range(dataset_size):
        if single:
            ratio = random.uniform(0.2, 2.0) if noise else 0
            signal_fq = random.randrange(10, 100, 10)
            signal_amp = random.randrange(10, 100, 10)
            signal, _, _ = factory.make_signal(signal_size=signal_size,
                                               impulse_amp=signal_amp,
                                               impulse_fq=signal_fq,
                                               noise_amp=signal_amp * ratio)
            X.append(signal)
            y.append([signal_fq])
        else:
            max_impulses = 3
            num_impulses = random.randint(1, max_impulses)
            signal_amp_fq = []
            for i in range(max_impulses):
                if i + 1 <= num_impulses:
                    fq = random.randrange(10, 100, 10)
                    amp = random.randrange(10, 100, 10)
                    signal_amp_fq.append((amp, fq))
                else:
                    # pad frequencies to be always to max_impulses
                    signal_amp_fq.append((0, 0))

            ratio = random.uniform(0.2, 2.0) if noise else 0
            noise_amp = random.choice([amp for (amp, fq) in signal_amp_fq if amp > 0]) * ratio
            signal, _, _ = factory.make_complex_signal(signal_size=signal_size,
                                                       impulses_amp_fq=signal_amp_fq,
                                                       noise_amp=noise_amp)
            X.append(signal)
            y.append([fq for (amp, fq) in signal_amp_fq])

    with open(path, 'w') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(["signal", "frequency"])

        for signal, fqs in zip(X, y):
            csv_writer.writerow([signal, fqs])


def read(path):
    """
        Reads dataset of signals and periodicities from a CSV and returns it as np.arrays

        Args:
            - path
                Path to read dataset from
        Returns: None
            - X
                Signals read from dataset as an np.array
            - y
                Corresponding periodicities read from dataset as an np.array

    """

    X = []
    y = []
    with open(path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader)  # skip header
        for row in csv_reader:
            X.append(np.array(literal_eval(row[0])))
            y.append(np.array(literal_eval(row[1])))

    return np.array(X), np.array(y)


def main():
    """ Main block """
    # write("{}/res/datasets/signals/single_clean_1000_1000.csv".format(rootpath.detect()),
    #       dataset_size=1000, signal_size=1000, single=True, noise=False)
    # write("{}/res/datasets/signals/single_noise_1000_1000.csv".format(rootpath.detect()),
    #       dataset_size=1000, signal_size=1000, single=True, noise=True)
    #
    # write("{}/res/datasets/signals/multiple_clean_1000_1000.csv".format(rootpath.detect()),
    #       dataset_size=1000, signal_size=1000, single=False, noise=False)
    # write("{}/res/datasets/signals/multiple_noise_1000_1000.csv".format(rootpath.detect()),
    #       dataset_size=1000, signal_size=1000, single=False, noise=True)

    write("{}/res/datasets/signals/single_clean_10000_1000.csv".format(rootpath.detect()),
          dataset_size=10000, signal_size=1000, single=True, noise=False)
    write("{}/res/datasets/signals/single_noise_10000_1000.csv".format(rootpath.detect()),
          dataset_size=10000, signal_size=1000, single=True, noise=True)

    write("{}/res/datasets/signals/multiple_clean_10000_1000.csv".format(rootpath.detect()),
          dataset_size=10000, signal_size=1000, single=False, noise=False)
    write("{}/res/datasets/signals/multiple_noise_10000_1000.csv".format(rootpath.detect()),
          dataset_size=10000, signal_size=1000, single=False, noise=True)

    # write("{}/res/datasets/signals/single_clean_10000_10000.csv".format(rootpath.detect()),
    #       dataset_size=10000, signal_size=10000, single=True, noise=False)
    # write("{}/res/datasets/signals/single_noise_10000_10000.csv".format(rootpath.detect()),
    #       dataset_size=10000, signal_size=10000, single=True, noise=True)
    #
    # write("{}/res/datasets/signals/multiple_clean_10000_10000.csv".format(rootpath.detect()),
    #       dataset_size=10000, signal_size=10000, single=False, noise=False)
    # write("{}/res/datasets/signals/multiple_noise_10000_10000.csv".format(rootpath.detect()),
    #       dataset_size=10000, signal_size=10000, single=False, noise=True)


if __name__ == '__main__':
    main()
