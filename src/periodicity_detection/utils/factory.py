""" Data factory """

import random
import math


def make_signal(signal_size=1000, impulse_amp=10, impulse_fq=10, noise_amp=100):
    """
        Generates data with noise and impulse on a set interval and plot generated data
    """
    impulse = [0] * signal_size
    for i in range(signal_size):
        if ((i % impulse_fq) == 0):
            if i == 0:
                impulse[i] = impulse_amp
                impulse[i + 1] = impulse_amp
            else:
                impulse[i] = impulse_amp
                impulse[i - 1] = impulse_amp
                impulse[i + 1] = impulse_amp

    noise = []
    for i in range(signal_size):
        n = random.uniform(0, noise_amp)
        noise.append(n)

    signal = [impulse[i] + noise[i] for i in range(signal_size)]
    return signal, impulse, noise


def make_complex_signal(signal_size=1000, impulses_amp_fq=[(10, 20), (15, 50)], noise_amp=100):
    """
        Generate data with noise and impulse on a set interval and plot generated data
    """
    signal = [0] * signal_size
    impulses = []
    for (impulse_amp, impulse_fq) in impulses_amp_fq:
        if impulse_amp > 0 and impulse_fq > 0:
            impulse = [0] * signal_size
            for i in range(signal_size):
                if ((i % impulse_fq) == 0):
                    if i == 0:
                        impulse[i] = impulse_amp
                        impulse[i + 1] = impulse_amp
                    else:
                        impulse[i] = impulse_amp
                        impulse[i - 1] = impulse_amp
                        impulse[i + 1] = impulse_amp

            impulses.append(impulse)
            for i in range(signal_size):
                signal[i] += impulse[i]

    noise = []
    for i in range(signal_size):
        n = random.uniform(0, noise_amp)
        signal[i] += n
        noise.append(n)

    return signal, impulses, noise


def make_cos_signal(frequency=1000, amplitude=1):
    """
        Generate clean cos data
    """
    dcoffset = 0
    t = numpy.arange(0, 0.00512, 0.00001)
    return [dcoffset + amplitude * math.cos(2 * math.pi * frequency * x) for x in t]
