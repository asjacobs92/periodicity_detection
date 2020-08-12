""" Plotter utils """

import math
import matplotlib.pyplot as plt
import itertools


def plot_signal(signal, impulse, noise, filename=None):
    """
        Plots signal information, as well as impulse and noise that form it
    """
    plt.subplot(3, 1, 1, title="Impulses")
    plt.plot(impulse)
    plt.subplot(3, 1, 2, title="Noise")
    plt.plot(noise)
    plt.subplot(3, 1, 3, title="Signal")
    plt.plot(signal)

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_multiple_signals(signal, impulses, noise, filename=None):
    """
        Plots signal information, as well as multiple impulses and noise that form it
    """
    plt.subplot(3, 1, 1, title="Impulse")
    for impulse in impulses:
        plt.plot(impulse)

    plt.subplot(3, 1, 2, title="Noise")
    plt.plot(noise)
    plt.subplot(3, 1, 3, title="Signal")
    plt.plot(signal)

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_coefficients(coeffs_by_level, filename=None):
    """
        Plots filtered signal for each level as processed, either hp or lp
    """
    plt.figure()

    for idx, level in enumerate(coeffs_by_level):
        plt.subplot(len(coeffs_by_level), 1, idx + 1)
        # plt.text(1.1, 0.5, "Level {}".format(idx), verticalalignment='center')
        plt.title("Lvl {}".format(idx), loc="right", y=0.1, x=1.12)
        plt.plot(coeffs_by_level[level])

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_time_series(time_series, filename=None):
    """
        Plots time series
    """
    plt.figure()
    plt.ylabel('# of packets')
    plt.xlabel('time (s)')
    plt.plot(time_series)

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_pdm_by_level(pdm_by_level, filename=None):
    """
        Plots  PDM analysis over each level of processed signal
    """

    plt.figure(facecolor='white')
    plt.title("Result of PDM analysis")
    plt.xlabel("Frequency")
    plt.ylabel("Theta")

    for idx, level in enumerate(pdm_by_level):
        plt.subplot(len(pdm_by_level), 1, idx + 1, title="Level {}".format(idx))
        period, theta = pdm_by_level[level]
        plt.plot(period, theta)

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_heatmap(hp_list, lp_list, title):
    """
        Generates heatmaps for each filter level as processed
    """

    if not hp_list or not lp_list:
        return

    ls = len(lp_list) - 1
    hs = len(hp_list)

    result = []
    max_len = len(hp_list[1])
    for i in range(1, hs + 1):
        fin_lev = []
        lev_len = len(hp_list[i])
        scale = math.floor(max_len / lev_len)
        rem = max_len % lev_len
        for j in range(rem):
            fin_lev.append(0)
        y = list(itertools.chain.from_iterable(itertools.repeat(x, int(scale)) for x in hp_list[i]))
        fin_lev += y
        result.append(fin_lev)

    fin_lev = []
    lev_len = len(lp_list[ls])
    scale = math.floor(max_len / lev_len)
    rem = max_len % lev_len
    for j in range(rem):
        fin_lev.append(0)
    y = list(itertools.chain.from_iterable(itertools.repeat(x, int(scale)) for x in lp_list[ls]))
    fin_lev += y
    result.append(fin_lev)

    plt.figure()
    plt.title(title)
    plt.imshow(result, interpolation='nearest', aspect='auto')
    plt.show()
