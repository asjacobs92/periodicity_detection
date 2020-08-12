"""  Main app """

import csv
import pywt
import math
import numpy as np

import PyAstronomy.pyTiming.pyPDM as py_pdm

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from transforms import dwt, swt, dwt_int
from utils import factory, plotter


def wavelet_test():
    """ Tests wavelets implementation against library """
    print("---------------- Start ----------------")
    # signal, impulse, noise = factory.make_signal(10, 100, 100)
    # plotter.plot_signal(signal, impulse, noise)

    signal = factory.make_cos_signal()

    [(cA2, cD2), (cA1, cD1)] = pywt.swt(signal, 'db1', level=2)

    hp_list_pywt = {
        0: signal,
        1: cD1,
        2: cD2,
    }

    lp_list_pywt = {
        0: signal,
        1: cA1,
        2: cA2,
    }

    hp_list, lp_list = swt(signal, 'db1', levels=2)

    for i in range(1, min(len(hp_list), len(hp_list_pywt))):
        print("HP Length: ", i, len(hp_list[i]), len(hp_list_pywt[i]))
        print("LP Length: ", i, len(lp_list[i]), len(lp_list_pywt[i]))

        is_equal_hp = True
        for d1, d2 in zip(hp_list[i], hp_list_pywt[i]):
            is_equal_hp &= math.isclose(d1, d2)
            print("Equal hp level?", i, math.isclose(d1, d2), d1, d2)

        print("Equal hp level?", i, is_equal_hp)

        is_equal_lp = True
        for d1, d2 in zip(hp_list[i], hp_list_pywt[i]):
            is_equal_lp &= math.isclose(d1, d2)
            print("Equal lp level?", i, math.isclose(d1, d2), d1, d2)

        print("Equal lp level?", i, is_equal_lp)

    plotter.plot_coefficients(hp_list_pywt, "High pass filter (detail level)")
    plotter.plot_coefficients(hp_list, "High pass filter (detail level)")

    plotter.plot_coefficients(lp_list_pywt, "Low pass filter (approximation level)")
    plotter.plot_coefficients(lp_list, "Low pass filter (approximation level)")

    print("----------------- End -----------------")


def ratio_test():
    """ Runs wavelets-pdm test with different signal to noise ratios """
    print("---------------- Start Ratio Test ----------------")

    impulse_amp, impulse_fq = 10, 50

    with open('../res/ratio/results.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['ratio', 'index', 'transform', 'level', 'filter', 'impulse fq',
                         'impulse amp', 'detected period', 'theta', 'detected impulse?'])

        for ratio in range(1, 11):
            for index in range(30):
                # generate signal
                signal, impulse, noise = factory.make_signal(impulse_amp, impulse_fq, impulse_amp * ratio)

                # apply wavelet transform
                dwt_hp_list, dwt_lp_list = dwt(signal, 'haar', levels=5)
                swt_hp_list, swt_lp_list = swt(signal, 'haar', levels=5)

                # apply pdm to each detail level
                bin_size = 10
                dwt_hp_pdm_by_level = {}
                for level, coeffs in dwt_hp_list.items():
                    pdm_scanner = py_pdm.Scanner(minVal=10, maxVal=100, dVal=1, mode="period")
                    if len(coeffs) >= 10:
                        pdm = py_pdm.PyPDM(np.asarray(range(len(coeffs))), np.asarray(coeffs))
                        period, theta = pdm.pdmEquiBin(bin_size, pdm_scanner)
                        dwt_hp_pdm_by_level[level] = (period, theta)
                        indexes_min = theta.argsort()[:3]
                        detected_periods = []
                        detected_periods_thetas = []
                        for idx in indexes_min:
                            detected_periods.append(period[idx])
                            detected_periods_thetas.append(theta[idx])

                        impulse_detected = 1 if impulse_fq in detected_periods else 0

                        writer.writerow([ratio, index, 'dwt', level, 'hp', impulse_amp, impulse_fq,
                                         detected_periods, detected_periods_thetas, impulse_detected])

                dwt_lp_pdm_by_level = {}
                for level, coeffs in dwt_lp_list.items():
                    pdm_scanner = py_pdm.Scanner(minVal=10, maxVal=100, dVal=1, mode="period")
                    if len(coeffs) >= 10:
                        pdm = py_pdm.PyPDM(np.asarray(range(len(coeffs))), np.asarray(coeffs))
                        period, theta = pdm.pdmEquiBin(bin_size, pdm_scanner)
                        dwt_lp_pdm_by_level[level] = (period, theta)
                        indexes_min = theta.argsort()[:3]
                        detected_periods = []
                        detected_periods_thetas = []
                        for idx in indexes_min:
                            detected_periods.append(period[idx])
                            detected_periods_thetas.append(theta[idx])

                        impulse_detected = 1 if impulse_fq in detected_periods else 0

                        writer.writerow([ratio, index, 'dwt', level, 'lp', impulse_amp, impulse_fq,
                                         detected_periods, detected_periods_thetas, impulse_detected])

                swt_hp_pdm_by_level = {}
                for level, coeffs in swt_hp_list.items():
                    pdm_scanner = py_pdm.Scanner(minVal=10, maxVal=100, dVal=bin_size, mode="period")
                    if len(coeffs) >= 10:
                        pdm = py_pdm.PyPDM(np.asarray(range(len(coeffs))), np.asarray(coeffs))
                        period, theta = pdm.pdmEquiBin(bin_size, pdm_scanner)
                        swt_hp_pdm_by_level[level] = (period, theta)
                        indexes_min = theta.argsort()[:3]
                        detected_periods = []
                        detected_periods_thetas = []
                        for idx in indexes_min:
                            detected_periods.append(period[idx])
                            detected_periods_thetas.append(theta[idx])

                        impulse_detected = 1 if impulse_fq in detected_periods else 0

                        writer.writerow([ratio, index, 'swt', level, 'hp', impulse_amp, impulse_fq,
                                         detected_periods, detected_periods_thetas, impulse_detected])
                swt_lp_pdm_by_level = {}
                for level, coeffs in swt_lp_list.items():
                    pdm_scanner = py_pdm.Scanner(minVal=10, maxVal=100, dVal=bin_size, mode="period")
                    if len(coeffs) >= 10:
                        pdm = py_pdm.PyPDM(np.asarray(range(len(coeffs))), np.asarray(coeffs))
                        period, theta = pdm.pdmEquiBin(bin_size, pdm_scanner)
                        swt_lp_pdm_by_level[level] = (period, theta)
                    indexes_min = theta.argsort()[:3]
                    detected_periods = []
                    detected_periods_thetas = []
                    for idx in indexes_min:
                        detected_periods.append(period[idx])
                        detected_periods_thetas.append(theta[idx])

                    impulse_detected = 1 if impulse_fq in detected_periods else 0

                    writer.writerow([ratio, index, 'swt', level, 'lp', impulse_amp, impulse_fq,
                                     detected_periods, detected_periods_thetas, impulse_detected])

            # plot last iteration only
            plotter.plot_signal(signal, impulse, noise,
                                filename="../res/ratio/signal_{}_{}_{}.png".format(impulse_amp, impulse_fq, impulse_amp * ratio))

            plotter.plot_coefficients(dwt_hp_list, filename="../res/ratio/dwt/coeffs_hp_ratio_{}.png".format(ratio))
            plotter.plot_coefficients(dwt_lp_list, filename="../res/ratio/dwt/coeffs_lp_ratio_{}.png".format(ratio))
            plotter.plot_coefficients(swt_hp_list, filename="../res/ratio/swt/coeffs_hp_ratio_{}.png".format(ratio))
            plotter.plot_coefficients(swt_lp_list, filename="../res/ratio/swt/coeffs_lp_ratio_{}.png".format(ratio))

            plotter.plot_pdm_by_level(
                dwt_hp_pdm_by_level, filename="../res/ratio/pdm/dwt_hp_pdm_ratio_{}.png".format(ratio))
            plotter.plot_pdm_by_level(
                dwt_lp_pdm_by_level, filename="../res/ratio/pdm/dwt_lp_pdm_ratio_{}.png".format(ratio))

            plotter.plot_pdm_by_level(
                swt_hp_pdm_by_level, filename="../res/ratio/pdm/swt_hp_pdm_ratio_{}.png".format(ratio))
            plotter.plot_pdm_by_level(
                swt_lp_pdm_by_level, filename="../res/ratio/pdm/swt_lp_pdm_ratio_{}.png".format(ratio))

    print("----------------- End Ratio Test -----------------")


def frequency_test():
    """ Runs wavelets-pdm test with different single to noise ratios """
    print("---------------- Start Frenquency Test ----------------")

    # (amp, fq)
    impulses_info = [(10, 10), (10, 30), (20, 75)]
    noise_amp = 20

    with open('../res/frequency/results.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['index', 'impulses', 'level', 'filter',
                         'detected periods', 'theta', 'num detected impulses'])

        for index in range(30):
            # generate signal
            signal, impulses, noise = factory.make_complex_signal(impulses_info, noise_amp)
            plotter.plot_multiple_signals(signal, impulses, noise, filename="../res/frequency/signal_multiple.png")

            # apply wavelet transform
            hp_list, lp_list = swt(signal, 'haar', levels=5)

            plotter.plot_coefficients(hp_list, filename="../res/frequency/coeffs_hp_ratio.png")
            plotter.plot_coefficients(lp_list, filename="../res/frequency/coeffs_lp_ratio.png")

            # apply pdm to each detail level
            bin_size = 50
            lp_pdm_by_level = {}
            for level, coeffs in lp_list.items():
                pdm_scanner = py_pdm.Scanner(minVal=10, maxVal=300, dVal=1, mode="period")
                if len(coeffs) >= 10:
                    pdm = py_pdm.PyPDM(np.asarray(range(len(coeffs))), np.asarray(coeffs))
                    period, theta = pdm.pdmEquiBin(bin_size, pdm_scanner)
                    lp_pdm_by_level[level] = (period, theta)
                    indexes_min = theta.argsort()[:3]
                    detected_periods = []
                    detected_periods_thetas = []
                    for idx in indexes_min:
                        detected_periods.append(period[idx])
                        detected_periods_thetas.append(theta[idx])

                    impulse_detected = 0
                    for (imp_amp, imp_fq) in impulses_info:
                        impulse_detected += 1 if imp_fq in detected_periods else 0

                    writer.writerow([index, [fq for (amp, fq) in impulses_info], level, 'lp',
                                     detected_periods, detected_periods_thetas, impulse_detected])

            hp_pdm_by_level = {}
            for level, coeffs in hp_list.items():
                pdm_scanner = py_pdm.Scanner(minVal=10, maxVal=300, dVal=1, mode="period")
                if len(coeffs) >= 10:
                    pdm = py_pdm.PyPDM(np.asarray(range(len(coeffs))), np.asarray(coeffs))
                    period, theta = pdm.pdmEquiBin(bin_size, pdm_scanner)
                    hp_pdm_by_level[level] = (period, theta)
                    indexes_min = theta.argsort()[:3]
                    detected_periods = []
                    detected_periods_thetas = []
                    for idx in indexes_min:
                        detected_periods.append(period[idx])
                        detected_periods_thetas.append(theta[idx])

                    impulse_detected = 0
                    for (imp_amp, imp_fq) in impulses_info:
                        impulse_detected += 1 if imp_fq in detected_periods else 0

                    writer.writerow([index, [fq for (amp, fq) in impulses_info], level, 'hp',
                                     detected_periods, detected_periods_thetas, impulse_detected])

            plotter.plot_pdm_by_level(lp_pdm_by_level, filename="../res/frequency/lp_pdm_.png")
            plotter.plot_pdm_by_level(hp_pdm_by_level, filename="../res/frequency/hp_pdm_.png")

    print("----------------- End Frenquency Test -----------------")


def autocorrelation_test():
    """ Tests autocorrelation function for periodictiy detection """

    # (amp, fq)
    impulses_info = [(10, 10), (20, 20)]
    noise_amp = 100

    # generate signal
    signal, impulses, noise = factory.make_complex_signal(impulses_info, noise_amp)

    # apply wavelet transform
    hp_list, lp_list = swt(signal, 'haar', levels=10)

    # apply pdm to each detail level
    autocorr_by_level = {}
    for level, coeffs in hp_list.items():
        result = np.correlate(coeffs, coeffs, mode='same')
        autocorr_by_level[level] = result[result.size // 2:]

    plotter.plot_coefficients(autocorr_by_level)


def int_error_test():
    """ Runs two implementations of the wavelets transform, floating points vs integers,
        to evaluate how much error is introduced. """

    impulse_amp, impulse_fq = 1000, 100
    pad = False

    # signal = factory.make_cos_signal()
    signal, impulse, noise = factory.make_signal(impulse_amp, impulse_fq, impulse_amp * 2)

    hp_list, lp_list = dwt(signal, 'haar', levels=10, pad=pad)
    hp_list_int, lp_list_int = dwt_int(signal, 'haar', levels=10, pad=pad)

    for i in range(1, min(len(hp_list), len(hp_list_int))):
        print("----- LEVEL {} START -----".format(i))

        print("HP LEVEL (FLOAT) {}".format(i), list(hp_list[i]))
        print("HP LEVEL (INT) {}".format(i), list(hp_list_int[i]))

        # is_equal_hp = True
        # for d1, d2 in zip(hp_list[i], hp_list_int[i]):
        #     is_equal_hp &= math.isclose(d1, d2)
        #     print("Equal hp level?", i, math.isclose(d1, d2), d1, d2)

        print("MSE of HP level {}:".format(i), mean_squared_error(hp_list[i], hp_list_int[i]))
        print("MAE of HP level {}:".format(i), mean_absolute_error(hp_list[i], hp_list_int[i]))
        print("R2 of HP level {}:".format(i), r2_score(hp_list[i], hp_list_int[i]))

        print("LP LEVEL (FLOAT) {}".format(i), list(lp_list[i]))
        print("LP LEVEL (INT) {}".format(i), list(lp_list_int[i]))

        # is_equal_lp = True
        # for d1, d2 in zip(hp_list[i], hp_list_int[i]):
        #     is_equal_lp &= math.isclose(d1, d2)
        #     print("Equal lp level?", i, math.isclose(d1, d2), d1, d2)

        print("MSE of LP level {}:".format(i), mean_squared_error(lp_list[i], lp_list_int[i]))
        print("MAE of LP level {}:".format(i), mean_absolute_error(lp_list[i], lp_list_int[i]))
        print("R2 of HP level {}:".format(i), r2_score(lp_list[i], lp_list_int[i]))

        print("----- LEVEL {} END -----".format(i))

    if pad:
        print("Total MSE of HP:", mean_squared_error(list(hp_list.values()), list(hp_list_int.values())))
        print("Total MAE of HP:", mean_absolute_error(list(hp_list.values()), list(hp_list_int.values())))
        print("Total R2 of HP:", r2_score(list(hp_list.values()), list(hp_list_int.values())))

        print("Total MSE of LP:", mean_squared_error(list(lp_list.values()), list(lp_list_int.values())))
        print("Total MAE of LP:", mean_absolute_error(list(lp_list.values()), list(lp_list_int.values())))
        print("Total R2 of LP:", r2_score(list(lp_list.values()), list(lp_list_int.values())))

    plotter.plot_coefficients(hp_list_int, filename="../res/int_error/hp_int.png")
    plotter.plot_coefficients(hp_list, filename="../res/int_error/hp_float.png")

    plotter.plot_coefficients(lp_list_int, filename="../res/int_error/lp_int.png")
    plotter.plot_coefficients(lp_list, filename="../res/int_error/lp_float.png")

    print("----------------- End -----------------")


def main():
    """ Runs wavelets-pdm test """
    print("---------------- Start ----------------")
    # wavelet_test()
    # ratio_test()
    # frequency_test()
    # autocorrelation_test()
    int_error_test()
    print("----------------- End -----------------")


if __name__ == '__main__':
    main()
