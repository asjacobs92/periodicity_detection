
import math
import time
import threading

import numpy as np
import PyAstronomy.pyTiming.pyPDM as py_pdm
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from transforms import swt, dwt
from utils import factory, plotter, parser


def seconds_to_scale(t): return int(math.log2(t))


buffer_variance = []
buffer_wavelets = []

hp_list = {}
lp_list = {}

variance = {}
log_variance = {}
sum_of_squares = {}

num_levels = 10
min_buffer = 2 ** num_levels

fig = plt.figure()
subplots = []
for level in range(num_levels + 1):
    hp_list[level] = []
    lp_list[level] = []
    sum_of_squares[level] = 0
    subplots.append(fig.add_subplot(math.ceil((num_levels + 1) / 2), 2, level + 1, title="Level {}".format(level)))

fig2 = plt.figure()
ax1 = fig2.add_subplot(111)
ax2 = ax1.twiny()

ax1.set_xlabel('Wavelet scale')
ax1.set_xticks(range(num_levels + 1))

# Decide the ticklabel position in the new x-axis,
# then convert them to the position in the old x-axis
new_labels = [2**x for x in range(num_levels + 1)]
new_pos = [seconds_to_scale(t) for t in new_labels]

ax2.set_xticks(new_pos)
ax2.set_xticklabels(new_labels)
ax2.set_xlabel('Time unit (s)')
ax2.set_xlim(ax1.get_xlim())

lock = threading.Lock()


def animate_variance(i):
    """ Checks if buffer is full to run wavelets and plot """
    global buffer_variance
    global min_buffer
    global hp_list, lp_list
    global sum_of_squares, variance, log_variance

    print('Animation variance', len(buffer_wavelets), min_buffer)
    if len(buffer_variance) >= min_buffer:
        lock.acquire()
        try:
            buffer_hp_list, buffer_lp_list = dwt(buffer_variance, 'haar', levels=num_levels)
            for idx, (level, coeffs) in enumerate(buffer_hp_list.items()):
                sum_of_squares[level] += sum([(x**2) for x in coeffs])
                variance[level] = sum_of_squares[level] / len(coeffs)
                log_variance[level] = math.log2(variance[level])

            buffer_variance = []  # clear buffer
            variance_x = list(log_variance.keys())
            X = np.reshape(variance_x, (len(variance_x), 1))
            y = list(log_variance.values())
            reg = LinearRegression()
            reg.fit(X, y)

            ax1.clear()
            ax1.set_xticks(range(num_levels + 1))
            ax1.plot(variance_x, y)
            ax1.plot(reg.predict(X))

        finally:
            lock.release()


def animate_wavelets(i):
    """ Checks if buffer is full to run wavelets and plot """
    global buffer_wavelets
    global subplots
    global min_buffer
    global hp_list, lp_list

    print('Animation wavelets', len(buffer_wavelets), min_buffer)
    if len(buffer_wavelets) >= min_buffer:
        lock.acquire()
        try:
            buffer_hp_list, buffer_lp_list = dwt(buffer_wavelets, 'haar', levels=num_levels)
            for idx, (level, coeffs) in enumerate(buffer_hp_list.items()):
                # hp_list[level] = hp_list[level][int(len(hp_list[level]) / 10):]
                hp_list[level] += coeffs

                ax = subplots[idx]
                ax.clear()
                ax.plot(hp_list[level])
            buffer_wavelets = []  # clear buffer
        finally:
            lock.release()


def read_packets():
    """ Reads pcap and adds packet count to buffer periodically """
    global buffer_wavelets, buffer_variance

    packets_per_second = parser.parse_pkts_per_second('../res/trickbot.pcap')

    for packet_count in packets_per_second:
        lock.acquire()
        try:
            buffer_wavelets.append(packet_count)
            buffer_variance.append(packet_count)
        finally:
            lock.release()
        time.sleep(0.001)


def real_time_test():
    """ Read pcap and reproduces signal to plot wavelets in real time """
    global fig, fig2
    thread = threading.Thread(target=read_packets)
    thread.start()
    anim = animation.FuncAnimation(fig, animate_wavelets, interval=10, frames=300)
    anim_2 = animation.FuncAnimation(fig2, animate_variance, interval=10, frames=300)
    plt.show()


def main():
    """ Main method """
    real_time_test()


if __name__ == '__main__':
    main()
