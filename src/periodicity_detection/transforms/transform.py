""" Module with wavelets transform implementation """
import numpy as np


from .wavelets import daubechies, haar


def get_filters(wavelet="haar"):
    """
        Fetches low pass and high pass filter for given scale and wavelet.
        Args:
            - scale
                Scale of the required filters.
            - wavelet
                Name of the required wavelet function.
                Currently supports: "haar", "daubechies_1", "daubechies_2", "daubechies_3", "daubechies_4".
                Default: "haar".
        Returns:
            * lp_filter
                Low pass filter vector (a.k.a. wavelet function or mother wavelet)
            * hp_filter
                High pass filter vector (a.k.a. scaling function or father wavelet)
    """

    wavelets = {
        "haar": haar(),
        "db1": daubechies(1),
        "db2": daubechies(2),
        "db3": daubechies(3),
        "db4": daubechies(4),
    }

    return wavelets.get(wavelet, haar)


def swt(signal, wavelet="haar", levels=2):
    """
        Applyies a discrete wavelet transform on the input signal.
        Args:
            - signal
                Vector of amplitude values per seconds consisting in input signal to be processed.
            - levels
                Maximum depth of signal decomposition.
                i.e., the number of the time the transform will be applied.
                Default: 2
            - wavelet
                Specifies which wavelet function and scaling function to use.
                Currently support: "haar", "db1", "db2", "db3", "db4".
                Default: "haar"
        Returns:
            * lp_list
                List of resulting signal points for each level of low pass filter.
                A.k.a. Approximation levels
            * hp_list
                List of resulting signal points for each level of high pass filter.
                A.k.a. Detail levels
    """
    return dwt(signal, wavelet, levels, decimated=False)


def dwt(signal, wavelet="haar", levels=2, decimated=True, pad=False):
    """
        Applyies a discrete wavelet transform on the input signal.
        Args:
            - signal
                Vector of amplitude values per seconds consisting in input signal to be processed.
            - levels
                Maximum depth of signal decomposition.
                i.e., the number of the time the transform will be applied.
                Default: 2
            - wavelet
                Specifies which wavelet function and scaling function to use.
                Currently support: "haar", "db1", "db2", "db3", "db4".
                Default: "haar"
        Returns:
            * lp_list
                List of resulting signal points for each level of low pass filter.
                A.k.a. Approximation levels
            * hp_list
                List of resulting signal points for each level of high pass filter.
                A.k.a. Detail levels
    """

    if(levels <= 0):
        raise ValueError("Invalid levels value")

    lp_filter, hp_filter = get_filters(wavelet)
    if lp_filter == None:
        raise ValueError("Invalid scale value")

    hp_list = {}
    lp_list = {}
    lp_list[0], hp_list[0] = signal, signal

    for level in range(1, levels + 1):
        lp_list[level], hp_list[level] = [], []

        # iterate through previous level to apply filter
        scale = len(lp_filter)
        for start in range(0, len(lp_list[level - 1]), 2 if decimated else 1):  # redundant dwt, no downsampling
            lp, hp = 0, 0
            for i in range(scale):
                # Applying filter to each point in the signal according to wavelet scale (convolution)
                if (start + i) < len(lp_list[level - 1]):
                    point = lp_list[level - 1][(start + i) % len(lp_list[level - 1])]
                    lp += point * lp_filter[scale - 1 - i]
                    hp += point * hp_filter[scale - 1 - i]

            lp_list[level].append(lp)
            hp_list[level].append(hp)

        if pad:
            lp_list[level] = list(np.pad(lp_list[level], (0, len(signal) - len(lp_list[level])),
                                         'constant', constant_values=(0, 0)))
            hp_list[level] = list(np.pad(hp_list[level], (0, len(signal) - len(hp_list[level])),
                                         'constant', constant_values=(0, 0)))

    return hp_list, lp_list


def dwt_int(signal, wavelet="haar", levels=2, decimated=True, pad=False):
    """
        Applyies a discrete wavelet transform on the input signal.
        Args:
            - signal
                Vector of amplitude values per seconds consisting in input signal to be processed.
            - levels
                Maximum depth of signal decomposition.
                i.e., the number of the time the transform will be applied.
                Default: 2
            - wavelet
                Specifies which wavelet function and scaling function to use.
                Currently support: "haar", "db1", "db2", "db3", "db4".
                Default: "haar"
        Returns:
            * lp_list
                List of resulting signal points for each level of low pass filter.
                A.k.a. Approximation levels
            * hp_list
                List of resulting signal points for each level of high pass filter.
                A.k.a. Detail levels
    """

    if(levels <= 0):
        raise ValueError("Invalid levels value")

    lp_filter, hp_filter = get_filters(wavelet)
    if lp_filter == None:
        raise ValueError("Invalid scale value")

    hp_list = {}
    lp_list = {}
    lp_list[0], hp_list[0] = signal, signal

    for level in range(1, levels + 1):
        lp_list[level], hp_list[level] = [], []

        # iterate through previous level to apply filter
        scale = len(lp_filter)
        for start in range(0, len(lp_list[level - 1]), 2 if decimated else 1):  # redundant dwt, no downsampling
            lp, hp = 0, 0
            for i in range(scale):
                # Applying filter to each point in the signal according to wavelet scale (convolution)
                if (start + i) < len(lp_list[level - 1]):
                    point = lp_list[level - 1][(start + i) % len(lp_list[level - 1])]
                    lp += int(point * lp_filter[scale - 1 - i])
                    hp += int(point * hp_filter[scale - 1 - i])

            lp_list[level].append(lp)
            hp_list[level].append(hp)

        if pad:
            lp_list[level] = list(np.pad(lp_list[level], (0, len(signal) - len(lp_list[level])),
                                         'constant', constant_values=(0, 0)))
            hp_list[level] = list(np.pad(hp_list[level], (0, len(signal) - len(hp_list[level])),
                                         'constant', constant_values=(0, 0)))

    return hp_list, lp_list
