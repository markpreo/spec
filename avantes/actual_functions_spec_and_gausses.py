import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import avaread
import matplotlib.patches as patches


def fillBkgs(datas, low_signal_times):
    bkgs = []
    for i in range(low_signal_times):
        bkgs.append(datas.scope.T[i])
    return bkgs


def gauss_Hb_big(x, A1, sigma_Hb, A0):
    return  A1 * np.exp(-(x - 485.908)**2 / (2 * sigma_Hb**2)) + A0
#486.137
def gauss_Db_big(x, A2, sigma_Db, A0):
    return  A2 * np.exp(-(x - 485.978)**2 / (2 * sigma_Db**2)) + A0
#486.045
def gauss_Hb_small(x, A3, sigma_Hs, A0):
    return  A3 * np.exp(-(x - 485.908)**2 / (2 * sigma_Hs**2)) + A0

def gauss_Db_small(x, A4, sigma_Ds, A0):
    return  A4 * np.exp(-(x - 485.978)**2 / (2 * sigma_Ds**2)) + A0

# def multi_gaussian(x, A1, A2, A3, A4, d_lambda, sigma_C, sigma_O, A0):
#     y_res_multi_g = A1 * np.exp(-(x - 464.74 - d_lambda)**2 / (2 * sigma_C**2)) +\
#                     A2 * np.exp(-(x - 464.916 - d_lambda) ** 2 / (2 * sigma_O ** 2)) + \
#                     A3 * np.exp(-(x - 465.025 - d_lambda) ** 2 / (2 * sigma_C ** 2)) + \
#                     A4 * np.exp(-(x - 465.147 - d_lambda) ** 2 / (2 * sigma_C ** 2)) + A0
#     return y_res_multi_g

def balmer_Hb_Db_gauss(x, A1, A2, sigma_Hb, sigma_Db, A0):
    y_res_balmer_Hb_Db_gauss = (A1 * np.exp(-(x - 485.908)**2 / (2 * sigma_Hb**2))) + (A2 * np.exp(-(x - 485.978)**2 / (2 * sigma_Db**2))) + A0
    return y_res_balmer_Hb_Db_gauss

def balmer_Hb_Db_gauss_4(x, A1, A2, A3, A4, sigma_Hb, sigma_Db, sigma_Hs, sigma_Ds, A0):
    y_res_balmer_Hb_Db_gauss = (A1 * np.exp(-(x - 485.908)**2 / (2 * sigma_Hb**2))) + (A2 * np.exp(-(x - 485.978)**2 / (2 * sigma_Db**2))) + \
                               (A3 * np.exp(-(x - 485.908)**2 / (2 * sigma_Hs**2))) + (A4 * np.exp(-(x - 485.978)**2 / (2 * sigma_Ds**2))) + A0
    return y_res_balmer_Hb_Db_gauss

#486.137
#486.03
def init_data(data_directory_1, data_directory_2):

    datas_blue = avaread.read_file(data_directory_1)

    low_signal_times = 3

    bkgs_b = fillBkgs(datas_blue, low_signal_times)
    bkgd_b = np.average(bkgs_b, axis=0)

    times = 15
    waves_b = datas_blue.wavelength

    if data_directory_2 != 0:
        datas_red = avaread.read_file(data_directory_2)
        bkgs_r = fillBkgs(datas_red, low_signal_times)
        bkgd_r = np.average(bkgs_r, axis=0)
        waves_r = datas_red.wavelength
        waves = np.append(waves_b, waves_r)
    else:
        waves = waves_b
        datas_red = 0
        bkgd_r = 0


    return waves, bkgd_b, bkgd_r, datas_blue, datas_red, times

def init_spectrum(data_directory_1, data_directory_2):
    waves, bkgd_b, bkgd_r, datas_blue, datas_red, times = init_data(data_directory_1, data_directory_2)
    final_spectrum = []
    for time in range(times):
        final_spectrum_b = datas_blue.scope.T[time] - bkgd_b
        if datas_red != 0:
            final_spectrum_r = datas_red.scope.T[time] - bkgd_r
            one_time_final_spectrum = np.append(final_spectrum_b, final_spectrum_r)
            one_time_final_spectrum = list(one_time_final_spectrum)
            final_spectrum.append(one_time_final_spectrum)
        else:
            one_time_final_spectrum = final_spectrum_b
            one_time_final_spectrum = list(one_time_final_spectrum)
            final_spectrum.append(one_time_final_spectrum)
    return final_spectrum, waves, times


def init_graph(data_directory_1, data_directory_2):
    plt.figure()
    colors = [
        '#0033CC',  # 1 - темно-синий (глубокий)
        '#1F4CFF',  # 2 - насыщенный синий
        '#3E6CFF',  # 3 - яркий синий
        '#1F8CFF',  # 4 - сине-голубой
        '#00A3CC',  # 5 - глубокий бирюзовый
        '#00B87A',  # 6 - изумрудно-зеленый
        '#00A350',  # 7 - темно-изумрудный
        '#2E8B57',  # 8 - морская волна (насыщенный)
        '#3C9E3C',  # 9 - глубокий зеленый
        '#5CAD2C',  # 10 - зеленый с желтым оттенком
        '#7CBC1C',  # 11 - оливково-зеленый
        '#9CCB0C',  # 12 - желто-зеленый
        '#CCCC00',  # 13 - глубокий желто-зеленый
        '#E6B800',  # 14 - золотисто-желтый
        '#FFCC00'  # 15 - глубокий желтый (золотой)
    ]

    linestyles = [
        'solid',  # 1
        'dashed',  # 2
        'dotted',  # 3
        'solid',  # 4
        'dashed',  # 5
        'dotted',  # 6
        'solid',  # 7
        'dashed',  # 8
        'dotted',  # 9
        'solid',  # 10
        'dashed',  # 11
        'dotted',  # 12
        'solid',  # 13
        'dashed',  # 14
        'dotted'  # 15
    ]
    final_spectrum, waves, times = init_spectrum(data_directory_1, data_directory_2)
    for time in range(times):
        plt.plot(waves, final_spectrum[time], color=colors[time], linestyle=linestyles[time], label=str(4 * time + 2) + ' ms')

    plt.xlabel('wavelenght, nm')
    plt.ylabel('Intesity, a. u.')
    plt.grid(True)
    plt.legend()
    plt.show()

def auto_peaking(chosen_time, data_directory_1, data_directory_2):
    final_spectrum, waves, times = init_spectrum(data_directory_1, data_directory_2)
    peaks = []
    for point in range(1, len(waves)-1):
        if (final_spectrum[chosen_time][point] > final_spectrum[chosen_time][point-1]) and \
            (final_spectrum[chosen_time][point] > final_spectrum[chosen_time][point+1]):
            peaks.append([waves[point], final_spectrum[chosen_time][point]])

    return peaks, final_spectrum, waves

def init_graph_autopeaking(chosen_time, i_filter, data_directory_1, data_directory_2):

    peaks, final_spectrum, waves = auto_peaking(chosen_time, data_directory_1, data_directory_2)

    plt.figure(figsize=(10, 6))
    plt.plot(waves, final_spectrum[chosen_time], 'b-', label='Экспериментальный спектр', alpha=0.7)

    for line in range(len(peaks)):
        if peaks[line][1] >= i_filter:
            plt.axvline(x=peaks[line][0], color='pink', linewidth=1)

    plt.xlabel('Длина волны (нм)')
    plt.ylabel('Интенсивность')
    plt.legend()
    plt.title('Autopeaking')
    plt.grid(True, alpha=0.3)

    plt.show()

def init_data_gauss(chosen_time, data_directory_1, data_directory_2):

    datas_blue = avaread.read_file(data_directory_1)
    spectrum_b = datas_blue.scope.T[chosen_time]
    waves_b = datas_blue.wavelength
    bkgd_b = fillBkgs(datas_blue, low_signal_times=3)

    if data_directory_2 != 0:
        datas_red = avaread.read_file(data_directory_2)
        spectrum_r = datas_red.scope.T[chosen_time]
        chosen_time_spectrum = np.array([])
        chosen_time_spectrum = np.append(spectrum_b, spectrum_r)
        waves_r = datas_red.wavelength
        waves = np.append(waves_b, waves_r)
        bkgd_r = fillBkgs(datas_red, low_signal_times=3)
        bkgd = np.append(bkgd_b, bkgd_r)
    else:
        waves = waves_b
        bkgd = bkgd_b
        chosen_time_spectrum = spectrum_b

    for i in range(len(chosen_time_spectrum)):
        chosen_time_spectrum[i] = chosen_time_spectrum[i] - bkgd[i]

    peak_region_mask = (waves >= 464.4) & (waves <= 465.6)
    x_region = waves[peak_region_mask]
    y_region = chosen_time_spectrum[peak_region_mask]

    A1, A2, A3, A4, d_lambda, sigma_C, sigma_O, A0 = 10370, 9490, 7600, 2950, 0, 0.1, 0.1, 0


    popt_full, pcov_full = curve_fit(
        multi_gaussian, x_region, y_region,
        p0=[A1, A2, A3, A4, d_lambda, sigma_C, sigma_O, A0]
    )

    return waves, chosen_time_spectrum, x_region, y_region, popt_full


def init_data_balmer_Hb_Db_gauss(chosen_time, data_directory_1, data_directory_2):

    datas_blue = avaread.read_file(data_directory_1)
    spectrum_b = datas_blue.scope.T[chosen_time]
    waves_b = datas_blue.wavelength
    bkgd_b = fillBkgs(datas_blue, low_signal_times=3)

    if data_directory_2 != 0:
        datas_red = avaread.read_file(data_directory_2)
        spectrum_r = datas_red.scope.T[chosen_time]
        chosen_time_spectrum = np.array([])
        chosen_time_spectrum = np.append(spectrum_b, spectrum_r)
        waves_r = datas_red.wavelength
        waves = np.append(waves_b, waves_r)
        bkgd_r = fillBkgs(datas_red, low_signal_times=3)
        bkgd_r = np.average(bkgd_r)
        bkgd_b = np.average(bkgd_b)
        bkgd = np.average(bkgd_b, bkgd_r)
    else:
        waves = waves_b
        bkgd = np.average(bkgd_b)
        chosen_time_spectrum = np.array(spectrum_b)


    print(bkgd)
    print(np.size(chosen_time_spectrum))
    for i in range(len(chosen_time_spectrum)):
        chosen_time_spectrum[i] = chosen_time_spectrum[i] - bkgd

    peak_region_mask = (waves >= 485) & (waves <= 487)
    x_region = waves[peak_region_mask]
    y_region = chosen_time_spectrum[peak_region_mask]

    A1, A2, A3, A4, sigma_Hb, sigma_Db, sigma_Hs, sigma_Ds, A0 = 5500, 16650, 0, 0, 0.1, 0.1, 0.5, 0.5, 0
    #A1, A2, sigma_Hb, sigma_Db, A0 = 17000, 42480, 0.1, 0.1, 0

    popt_full_1, pcov_full_1 = curve_fit(
        balmer_Hb_Db_gauss_4, x_region, y_region,
        p0=[A1, A2, A3, A4, sigma_Hb, sigma_Db, sigma_Hs, sigma_Ds, A0],
        bounds=(0, 1000000)
    )

    return waves, chosen_time_spectrum, x_region, y_region, popt_full_1



def init_graph_gauss(chosen_time, data_directory_1, data_directory_2):
    #waves, chosen_time_spectrum, x_region, y_region, popt_full = init_data_gauss(chosen_time, data_directory_1, data_directory_2)
    waves, chosen_time_spectrum, x_region, y_region, popt_full_1 = init_data_balmer_Hb_Db_gauss(chosen_time, data_directory_1, data_directory_2)
    # кривая для отображения
    x_fit = np.linspace(x_region.min(), x_region.max(), 500)
    y_fit = balmer_Hb_Db_gauss_4(x_fit, *popt_full_1)


    plt.figure(figsize=(10, 6))
    plt.plot(waves, chosen_time_spectrum, 'b-', label='Экспериментальный спектр', alpha=0.7)
    plt.plot(x_region, y_region, 'ro', label='Область аппроксимации', markersize=4)
    plt.plot(x_fit, y_fit, 'r-', linewidth=2)

    print(popt_full_1)

    y_popt = []
    y_popt.append(popt_full_1[0])
    y_popt.append(popt_full_1[4])
    y_popt.append(popt_full_1[8])
    print(y_popt)
    y_fit_Hb_big = gauss_Hb_big(x_fit, *y_popt)
    plt.plot(x_fit, y_fit_Hb_big, 'r', linestyle='dashed', linewidth=2, label='H_warm')

    y_popt_1 = []
    y_popt_1.append(popt_full_1[1])
    y_popt_1.append(popt_full_1[5])
    y_popt_1.append(popt_full_1[8])
    print(y_popt_1)
    y_fit_Db_big = gauss_Db_big(x_fit, *y_popt_1)
    plt.plot(x_fit, y_fit_Db_big, 'g', linestyle='dashed', linewidth=2,  label='D_warm')

    y_popt = []
    y_popt.append(popt_full_1[2])
    y_popt.append(popt_full_1[6])
    y_popt.append(popt_full_1[8])
    y_fit_Hb_small = gauss_Hb_small(x_fit, *y_popt)
    plt.plot(x_fit, y_fit_Hb_small, 'r', linestyle='dotted', linewidth=2,  label='H_hot')

    y_popt = []
    y_popt.append(popt_full_1[3])
    y_popt.append(popt_full_1[7])
    y_popt.append(popt_full_1[8])

    y_fit_Db_small = gauss_Db_small(x_fit, *y_popt)
    plt.plot(x_fit, y_fit_Db_small, 'g', linestyle='dotted', linewidth=2,  label='D_hot')

    plt.xlabel('Длина волны (нм)')
    plt.ylabel('Интенсивность')
    plt.legend()
    plt.title('Гауссова аппроксимация спектрального пика')
    plt.grid(True, alpha=0.3)

    plt.show()


def main():
    data_directory_1 = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\260428\1.STR8'
    #data_directory_1 = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\111225\p40 45.STR8'
    #data_directory_2 = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\111225\p40 66.STR8'
    #init_graph(data_directory_1, data_directory_2)

    #init_graph(data_directory_1, data_directory_2=0)
    chosen_time = 3
    i_filter = 300
    #init_graph_gauss(chosen_time, data_directory_1, data_directory_2=0)
    init_graph_autopeaking(chosen_time, i_filter, data_directory_1, data_directory_2=0)

main()

#
#
#
# peak_region_mask = (waves >= 464.83393573) & (waves <= 465.216)
# x_region = waves[peak_region_mask]
# print(x_region)
# y_region = chosen_time_spectrum[peak_region_mask]
# print(y_region)
#
#
#
# bkgd_1 = np.average(bkgd_b) # зависит от канала
# bkgd_2 = np.average(bkgd_r)
# bkgd = (bkgd_1 + bkgd_2)/2
# print(bkgd)
#
# popt_full, pcov_full = curve_fit(
#     gaussian,
#     x_region,
#     y_region,
#     p0=[amplitude, center, sigma, bkgd]
# )
#
#
# # Стандартные метрики
# FWHM = 2.355 * sigma  # полная ширина на половине высоты
# R_squared = 1 - np.sum((y_region - gaussian(x_region, *popt_full))**2) / np.sum((y_region - np.mean(y_region))**2)
#
# print(f"Центр пика: {center:.4f} нм")
# print(f"Амплитуда: {amplitude:.1f}")
# print(f"Sigma: {sigma:.4f} нм")
# print(f"FWHM: {FWHM:.4f} нм")
# print(f"Фон: {bkgd:.1f}")
# print(f"R²: {R_squared:.4f}")
#
#
# # Создаём гладкую кривую для отображения
# x_fit = np.linspace(x_region.min(), x_region.max(), 500)
# y_fit = gaussian(x_fit, *popt_full)
#
# plt.plot(waves, chosen_time_spectrum, 'b-', label='Экспериментальный спектр', alpha=0.7)
# plt.plot(x_region, y_region, 'ro', label='Область аппроксимации', markersize=4)
# plt.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Гаусс (центр = {center:.3f} нм)')
# plt.axvline(center, color='r', linestyle='--', alpha=0.5, label=f'Центр = {center:.3f} нм')
# plt.xlabel('Длина волны (нм)')
# plt.ylabel('Интенсивность')
# plt.legend()
# plt.title('Гауссова аппроксимация спектрального пика')
# plt.grid(True, alpha=0.3)
# plt.show()