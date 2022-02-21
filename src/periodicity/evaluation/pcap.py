
import math
import time
import numpy as np
import PyAstronomy.pyTiming.pyPDM as py_pdm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from transforms import swt, dwt
from utils import factory, plotter, parser


def per_millisecond_test():
    """ Analyses periodicity in signal broke down by packet per second """
    print("---------------- Start Per Millisecond Test ----------------")

    packets_per_millisecond = parser.parse_pkts_per_millisecond('../res/trickbot.pcap')
    num_splits = 100
    split_size = int(len(packets_per_millisecond) / num_splits)
    chunks = [packets_per_millisecond[x:x + split_size] for x in range(0, len(packets_per_millisecond), split_size)]

    for split, chunk in enumerate(chunks):
        print("---------------- Start Chunk {}/{} with {} ----------------".format(split, num_splits, split_size))
        plotter.plot_time_series(chunk, filename="../res/pcap/per_millisecond/time_series/time_series_{}".format(split))
        hp_list, lp_list = swt(chunk, 'haar', levels=10)

        plotter.plot_coefficients(hp_list, filename="../res/pcap/per_millisecond/coeffs/coeffs_{}".format(split))

        bin_size = 100
        pdm_by_level = {}
        for level, coeffs in hp_list.items():
            pdm_scanner = py_pdm.Scanner(minVal=100, maxVal=1000, dVal=100, mode="period")
            if len(coeffs) >= 10:
                pdm = py_pdm.PyPDM(np.asarray(range(len(coeffs))), np.asarray(coeffs))
                period, theta = pdm.pdmEquiBin(bin_size, pdm_scanner)
                pdm_by_level[level] = (period, theta)
                indexes_min = theta.argsort()[:3]
                detected_periods = []
                for idx in indexes_min:
                    detected_periods.append(period[idx])

                print('Detected periods', detected_periods)

        plotter.plot_pdm_by_level(pdm_by_level, filename="../res/pcap/per_millisecond/pdm/pdm_{}".format(split))

        print("---------------- End Chunk {}/{} with {} ----------------".format(split, num_splits, split_size))

    print("---------------- End Per Millisecond Test ----------------")


def per_second_test():
    """ Analyses periodicity in signal broke down by packet per second """
    print("---------------- Start Per Second Test ----------------")

    packets_per_second = parser.parse_pkts_per_second('../res/trickbot.pcap')
    num_splits = 10
    split_size = int(len(packets_per_second) / num_splits)
    chunks = [packets_per_second[x:x + split_size] for x in range(0, len(packets_per_second), split_size)]

    for split, chunk in enumerate(chunks):
        print("---------------- Start Chunk {}/{} with {} ----------------".format(split, num_splits, split_size))

        plotter.plot_time_series(chunk, filename="../res/pcap/per_second/time_series/time_series_{}".format(split))
        hp_list, lp_list = dwt(chunk, 'haar', levels=10)

        plotter.plot_coefficients(hp_list, filename="../res/pcap/per_second/coeffs/coeffs_{}".format(split))

        bin_size = 50
        pdm_by_level = {}
        for level, coeffs in hp_list.items():
            pdm_scanner = py_pdm.Scanner(minVal=10, maxVal=100, dVal=1, mode="period")
            if len(coeffs) >= 10:
                pdm = py_pdm.PyPDM(np.asarray(range(len(coeffs))), np.asarray(coeffs))
                period, theta = pdm.pdmEquiBin(bin_size, pdm_scanner)
                pdm_by_level[level] = (period, theta)
                indexes_min = theta.argsort()[:3]
                detected_periods = []
                for idx in indexes_min:
                    detected_periods.append(period[idx])

                print('Detected periods', detected_periods)

        plotter.plot_pdm_by_level(pdm_by_level, filename="../res/pcap/per_second/pdm/pdm_{}".format(split))
        print("---------------- End Chunk {}/{} with {} ----------------".format(split, num_splits, split_size))

    print("---------------- End Per Second Test ----------------")


def squared(n):
    """ Squares n in a streaming fashion """

    print(n, int(n))
    squared = 0
    x = 1
    for i in range(abs(int(n))):
        squared += x
        x += 2

    print('squared', squared)
    return squared


def sum_of_squares_test():
    """ Analyses sum of squares in signal broke down by packet per second """
    print("---------------- Start Per Second Test ----------------")

    packets_per_second = parser.parse_pkts_per_second('../res/trickbot.pcap')
    num_splits = 10
    split_size = int(len(packets_per_second) / num_splits)
    chunks = [packets_per_second[x:x + split_size] for x in range(0, len(packets_per_second), split_size)]
    num_levels = 10

    for split, chunk in enumerate(chunks):
        print("---------------- Start Chunk {}/{} with {} ----------------".format(split, num_splits, split_size))

        plotter.plot_time_series(chunk, filename="../res/pcap/sum_of_squares/time_series/time_series_{}".format(split))
        hp_list, lp_list = dwt(chunk, 'haar', levels=num_levels)

        plotter.plot_coefficients(hp_list, filename="../res/pcap/sum_of_squares/coeffs/coeffs_{}".format(split))

        variance = {}
        log_variance = {}
        sum_of_squares = {}
        for level, coeffs in hp_list.items():
            sum_of_squares[level] = sum([squared(x) for x in coeffs])
            variance[level] = sum_of_squares[level] / len(coeffs)
            print('variance', variance[level])
            log_variance[level] = 0 if variance[level] == 0 else math.log2(variance[level])

            print('sum_of_squares, variance', level, sum_of_squares[level], variance[level], log_variance[level])

        X = np.reshape(list(log_variance.keys()), (len(log_variance.keys()), 1))
        y = list(log_variance.values())
        reg = LinearRegression()
        reg.fit(X, y)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()

        ax1.set_ylabel('Variance')
        ax1.set_xlabel('Wavelet scale')
        ax1.set_xticks(range(num_levels + 1))
        ax1.plot(list(log_variance.keys()), list(log_variance.values()))
        ax1.plot(reg.predict(X))

        # Decide the ticklabel position in the new x-axis,
        # then convert them to the position in the old x-axis
        new_labels = [2**x for x in range(num_levels + 1)]

        def seconds_to_scale(t): return int(math.log2(t))
        new_pos = [seconds_to_scale(t) for t in new_labels]

        ax2.set_xticks(new_pos)
        ax2.set_xticklabels(new_labels)
        ax2.set_xlabel('Time unit (s)')
        ax2.set_xlim(ax1.get_xlim())

        plt.savefig("../res/pcap/sum_of_squares/variance/variance_{}".format(split))
        plt.close()

        print("---------------- End Chunk {}/{} with {} ----------------".format(split, num_splits, split_size))
    print("---------------- End Per Second Test ----------------")


def main():
    """ Main method """
    sum_of_squares_test()
    # per_second_test()
    # per_millisecond_test()


if __name__ == '__main__':
    main()
