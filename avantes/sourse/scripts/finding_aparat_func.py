import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from functools import partial
import avaread
import matplotlib.patches as patches
import math
import os
from scipy.stats import alpha
import pandas as pd
from sklearn.metrics import r2_score
from scipy.signal import resample
from matplotlib.colors import to_hex



def fillBkgs(datas, low_signal_times):
    bkgs = []
    for i in range(low_signal_times):
        bkgs.append(datas.scope.T[i])
    return bkgs


def init_data(data_directory_1, data_directory_2, data_background_1, data_background_2):
    global times, red_times, waves_b, waves_r, waves
    datas_blue = avaread.read_file(data_directory_1)

    times = len(datas_blue)

    waves_b = datas_blue.wavelength

    bkgd_b = avaread.read_file(data_background_1)

    if data_directory_2 != 0:
        datas_red = avaread.read_file(data_directory_2)

        red_times = len(datas_red)
        waves_r = datas_red.wavelength
        waves = np.append(waves_b, waves_r)
        bkgd_r = avaread.read_file(data_background_2)
    else:
        red_times = 0
        waves_r = 0
        waves = waves_b
        datas_red = 0
        bkgd_r = 0

    return bkgd_b, bkgd_r, datas_blue, datas_red



def init_spectrum(data_directory_1, data_directory_2, data_background_1, data_background_2):
    bkgd_b, bkgd_r, datas_blue, datas_red = init_data(data_directory_1, data_directory_2,
                                                             data_background_1, data_background_2)
    final_spectrum = np.empty((times + red_times), dtype=object)
    for time in range(times):
        final_spectrum[time] = datas_blue.scope.T[time] - bkgd_b.scope.T[time]

    if datas_red != 0:
        for rtime in range(times, times + red_times):
            final_spectrum[rtime] = datas_red.scope.T[rtime] - bkgd_r.scope.T[rtime]

    return final_spectrum


def averaging_spectrum(data_directory_1, data_directory_2, data_background_1, data_background_2):
    final_spectrum = init_spectrum(data_directory_1, data_directory_2, data_background_1, data_background_2)

    averaged_final_spectrum = np.empty((2), dtype=object)

    if red_times != 0:
        averaged_final_spectrum[0] = np.mean(final_spectrum[:times, :], axis=0)
        averaged_final_spectrum[1] = np.mean(final_spectrum[times:, :], axis=0)
    else:
        averaged_final_spectrum[0] = np.mean(final_spectrum, axis=0)

    return averaged_final_spectrum


def cutting_off_peaks(curr_waves, widths, data_directory_1, data_directory_2, data_background_1, data_background_2):
    averaged_final_spectrum = averaging_spectrum(data_directory_1, data_directory_2, data_background_1, data_background_2)

    peaks = np.empty((len(curr_waves), 2), dtype=object)
    print(peaks.shape)

    for wave in range(len(curr_waves)):
        if red_times == 0:
            print(wave)
            wi = widths[wave]
            peak_region_mask = (waves >= curr_waves[wave] - wi) & (waves <= curr_waves[wave] + wi)
            x_region = waves[peak_region_mask]
            y_region = averaged_final_spectrum[0][peak_region_mask]
            peaks[wave][0] = x_region
            peaks[wave][1] = y_region
        else:
            print(wave)
            wi = widths[wave]
            print(wi)
            peak_region_mask = (waves >= curr_waves[wave] - wi) & (waves <= curr_waves[wave] + wi)
            x_region = waves[peak_region_mask]
            y_region = averaged_final_spectrum[1][peak_region_mask]
            peaks[wave][0] = x_region
            peaks[wave][1] = y_region

    return peaks

# def get_size_coefs(curr_waves, chosen_time, widths, data_directory_1, data_directory_2, data_background_1, data_background_2):
#     peaks = cutting_off_peaks(curr_waves,
#                       chosen_time,
#                       widths,
#                       data_directory_1,
#                       data_directory_2,
#                       data_background_1,
#                       data_background_2)
#
#     size_coefs_each_peak = []
#     max_p_first_peak = max(peaks[0][1])
#     # созраняем коэффициенты
#     for i in range(len(peaks)):
#         curr_coef = max(peaks[i][1]) / max_p_first_peak
#         size_coefs_each_peak.append(curr_coef)
#         print(size_coefs_each_peak)
#
#     return size_coefs_each_peak, peaks
#
#
# def resizing_peaks(curr_waves, chosen_time, widths, data_directory_1, data_directory_2, data_background_1, data_background_2):
#     size_coefs_each_peak, peaks = get_size_coefs(curr_waves,
#                                                  chosen_time,
#                                                  widths,
#                                                  data_directory_1,
#                                                  data_directory_2,
#                                                  data_background_1,
#                                                  data_background_2)
#
#     resized_peaks = np.zeros_like(peaks)
#     for i in range(len(peaks)):
#         resized_peaks[i][0] = peaks[i][0]
#         resized_peaks[i][1] = peaks[i][1] / size_coefs_each_peak[i]
#
#     return resized_peaks


def resampling_peaks(curr_waves, widths, data_directory_1, data_directory_2, data_background_1, data_background_2):
    peaks = cutting_off_peaks(curr_waves, widths,
                              data_directory_1, data_directory_2, data_background_1, data_background_2)


    resampled_peaks = np.empty(
        peaks.shape,
        dtype=object
    )

    x_max_cord = np.zeros(len(peaks))
    y_max_cord = np.zeros(len(peaks))

    for i in range(len(resampled_peaks)):
        resampled_peaks[i][1], resampled_peaks[i][0] = resample(peaks[i][1], 300, t=peaks[i][0])
        # cord_mask = (resampled_peaks[i][1] == max(resampled_peaks[i][1]))
        max_indx = np.argmax(resampled_peaks[i][1])
        x_max_cord[i] = resampled_peaks[i][0][max_indx]
        y_max_cord[i] =  resampled_peaks[i][1][max_indx]

    return resampled_peaks, x_max_cord, y_max_cord

def resizing_peaks(curr_waves, widths, data_directory_1, data_directory_2, data_background_1, data_background_2):
    resampled_peaks, x_max_cord, y_max_cord = resampling_peaks(curr_waves, widths,
                                                              data_directory_1, data_directory_2, data_background_1, data_background_2)

    max_f_p = max(resampled_peaks[0][1])

    resized_resampled_peaks = np.empty(
        resampled_peaks.shape,
        dtype=object
    )

    for i in range(len(resampled_peaks)):
        coef = max(resampled_peaks[i][1]) / max_f_p
        resized_resampled_peaks[i][1] = resampled_peaks[i][1] / coef
        resized_resampled_peaks[i][0] = resampled_peaks[i][0]

    return resized_resampled_peaks, x_max_cord, y_max_cord


def init_graph_comparison(curr_waves, widths, data_directory_1, data_directory_2, data_background_1, data_background_2):
    resized_resampled_peaks, x_max_cord, y_max_cord = resizing_peaks(curr_waves, widths,
                                                                     data_directory_1, data_directory_2,
                                                                     data_background_1, data_background_2)

    x = np.arange(1, 301)
    plt.figure()

    n = 20

    colors = [
        to_hex(color)
        for color in plt.cm.rainbow(np.linspace(0, 1, n))
    ]


    for i in range(len(resized_resampled_peaks)):
        print(resized_resampled_peaks[i][0])
        for j in range(len(resized_resampled_peaks[i][0])):
            resized_resampled_peaks[i][0][j] = resized_resampled_peaks[i][0][j] - x_max_cord[i]
        plt.plot(resized_resampled_peaks[i][0], resized_resampled_peaks[i][1], color=colors[i], label=str(i))

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    # сравниваю максимумы для каждой группы у и нахожу соэффициенты отношений. к малому? или большому.
    # далее по хорошему надо ресэмплить их, чтобы получить больше точек. ну или значала ресэмплить потом сравнивать.
    # потом нормирую - каждую группу на коэф.
    # далее строю н графиков на одном окне и накладываю пики друг на друга





def auto_peaking(data_directory_1, data_directory_2, data_background_1, data_background_2):
    final_spectrum = averaging_spectrum(data_directory_1, data_directory_2,
                                                 data_background_1, data_background_2)
    peaks = []
    if red_times != 0:
        for point in range(1, len(waves)-1-len(waves_r)):
            if (final_spectrum[0][point] > final_spectrum[0][point-1]) and \
                (final_spectrum[0][point] > final_spectrum[0][point+1]):
                peaks.append([waves[point], final_spectrum[0][point]])

        for rpoint in range(len(waves_r)+1, len(waves)-1):
            if (final_spectrum[1][rpoint] > final_spectrum[1][rpoint-1]) and \
                (final_spectrum[1][rpoint] > final_spectrum[1][rpoint+1]):
                peaks.append([waves[rpoint], final_spectrum[1][rpoint]])
    else:
        for point in range(1, len(waves)-1):
            if (final_spectrum[0][point] > final_spectrum[0][point-1]) and \
                (final_spectrum[0][point] > final_spectrum[0][point+1]):
                peaks.append([waves[point], final_spectrum[0][point]])

    return peaks, final_spectrum


def init_graph_autopeaking(i_filter, data_directory_1, data_directory_2, data_background_1, data_background_2):

    peaks, final_spectrum = auto_peaking(data_directory_1, data_directory_2,
                                                data_background_1, data_background_2)

    plt.figure(figsize=(10, 6))

    if red_times != 0:
        plt.plot(waves, final_spectrum[0] + final_spectrum[1], 'b-', label='Экспериментальный спектр', alpha=0.7)
    else:
        plt.plot(waves, final_spectrum[0], 'b-', label='Экспериментальный спектр', alpha=0.7)

    final_peaks = []
    for line in range(len(peaks)):
        if peaks[line][1] >= i_filter:
            plt.axvline(x=peaks[line][0], color='pink', linewidth=1)
            final_peaks.append(peaks[line][0])

    plt.xlabel('Длина волны (нм)')
    plt.ylabel('Интенсивность')
    plt.legend()
    plt.title('Autopeaking')
    plt.grid(True, alpha=0.3)

    plt.show()
    return final_peaks


def main():
    data_directory_1 = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\data\260701\RED_neon_lamp_210v.STR8'
    data_background_1 = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\data\260701\RED_neon_lamp_background.STR8'
    data_directory_2 = 0
    data_background_2 = 0

    i_filter = 500
    peaks = init_graph_autopeaking(i_filter,
                           data_directory_1, data_directory_2,
                           data_background_1, data_background_2)

    print(peaks)
    curr_waves = peaks

    widths = [0.8 for i in range(len(curr_waves))]

    init_graph_comparison(curr_waves, widths,
                          data_directory_1, data_directory_2,
                          data_background_1, data_background_2)


main()