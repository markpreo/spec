import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import avaread
import matplotlib.patches as patches
import math
import os


def fillBkgs(datas, low_signal_times):
    bkgs = []
    for i in range(low_signal_times):
        bkgs.append(datas.scope.T[i])
    return bkgs


def gauss_Hb_big(x, A1, sigma_Hb, A0):
    return  A1 * np.exp(-(x - 486.135)**2 / (2 * sigma_Hb**2)) + A0
#486.137
def gauss_Db_big(x, A2, sigma_Db, A0):
    return  A2 * np.exp(-(x - 486.00013)**2 / (2 * sigma_Db**2)) + A0
#486.045
def gauss_Hb_small(x, A3, sigma_Hs, A0):
    return  A3 * np.exp(-(x - 486.135)**2 / (2 * sigma_Hs**2)) + A0

def gauss_Db_small(x, A4, sigma_Ds, A0):
    return  A4 * np.exp(-(x - 486.00013)**2 / (2 * sigma_Ds**2)) + A0

# def multi_gaussian(x, A1, A2, A3, A4, d_lambda, sigma_C, sigma_O, A0):
#     y_res_multi_g = A1 * np.exp(-(x - 464.74 - d_lambda)**2 / (2 * sigma_C**2)) +\
#                     A2 * np.exp(-(x - 464.916 - d_lambda) ** 2 / (2 * sigma_O ** 2)) + \
#                     A3 * np.exp(-(x - 465.025 - d_lambda) ** 2 / (2 * sigma_C ** 2)) + \
#                     A4 * np.exp(-(x - 465.147 - d_lambda) ** 2 / (2 * sigma_C ** 2)) + A0
#     return y_res_multi_g

def balmer_Hb_Db_gauss(x, A1, A2, sigma_Hb, sigma_Db, A0):
    y_res_balmer_Hb_Db_gauss = (A1 * np.exp(-(x - 486.135)**2 / (2 * sigma_Hb**2))) + (A2 * np.exp(-(x - 486.00013)**2 / (2 * sigma_Db**2))) + A0
    return y_res_balmer_Hb_Db_gauss

def balmer_Hb_Db_gauss_4(x, A1, A2, A3, A4, sigma_Hb, sigma_Db, sigma_Hs, sigma_Ds, A0):
    y_res_balmer_Hb_Db_gauss = (A1 * np.exp(-(x - 486.135)**2 / (2 * sigma_Hb**2))) + (A2 * np.exp(-(x - 486.00013)**2 / (2 * sigma_Db**2))) + \
                               (A3 * np.exp(-(x - 486.135)**2 / (2 * sigma_Hs**2))) + (A4 * np.exp(-(x - 486.00013)**2 / (2 * sigma_Ds**2))) + A0
    return y_res_balmer_Hb_Db_gauss

def gauss_Hg_big(x, A1, sigma_Hg, A0):
    return  A1 * np.exp(-(x - 434.0472)**2 / (2 * sigma_Hg**2)) + A0
#486.137
def gauss_Dg_big(x, A2, sigma_Dg, A0):
    return  A2 * np.exp(-(x - 433.92833)**2 / (2 * sigma_Dg**2)) + A0
#486.045
def gauss_Hg_small(x, A3, sigma_Hsg, A0):
    return  A3 * np.exp(-(x - 434.0472)**2 / (2 * sigma_Hsg**2)) + A0

def gauss_Dg_small(x, A4, sigma_Dsg, A0):
    return  A4 * np.exp(-(x - 433.92833)**2 / (2 * sigma_Dsg**2)) + A0


def balmer_Hg_Dg_gauss_4(x, A1, A2, A3, A4, sigma_Hg, sigma_Dg, sigma_Hsg, sigma_Dsg, A0):
    y_res_balmer_Hg_Dg_gauss = (A1 * np.exp(-(x - 434.0472)**2 / (2 * sigma_Hg**2))) + (A2 * np.exp(-(x - 433.92833)**2 / (2 * sigma_Dg**2))) + \
                               (A3 * np.exp(-(x - 434.0472)**2 / (2 * sigma_Hsg**2))) + (A4 * np.exp(-(x - 433.92833)**2 / (2 * sigma_Dsg**2))) + A0
    return y_res_balmer_Hg_Dg_gauss
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


def apparat_func_corrected(chosen_time, final_spectrum, waves):
    pass


def device_errors(data_directory_1, data_directory_2):
    data_blue = avaread.read_file(data_directory_1)
    errors_blue = []
    for i in range(len(data_blue)):
        errors_blue.append(data_blue.scope.T[i])

    #расчет сред кв отклон для каждого знач.
    res_std_blue = np.std(errors_blue, axis=0)
    print(res_std_blue)
    print(len(res_std_blue))


    if data_directory_2 != 0:
        data_red = avaread.read_file(data_directory_2)
        errors_red = []
        for i in range(len(data_red)):
            errors_red.append(data_red.scope.T[i])
        res_std_red = np.std(errors_red, axis=1)
        print(res_std_red)
        print(len(res_std_red))
    else:
        res_std_red = 0

    return res_std_blue, res_std_red

def init_data_std_errors_blue(res_std_blue):
    width_gap = 0.05
    first_std_blue = min(res_std_blue)
    count_of_bins = math.floor((max(res_std_blue) - min(res_std_blue)) / width_gap)
    print(count_of_bins)
    subs_gist = []
    height_gist = []
    for j in range(count_of_bins):
        print(first_std_blue)
        count = 0
        buff_one_gist = []
        for i in range(len(res_std_blue)):
            if (res_std_blue[i] <= first_std_blue + width_gap) and (res_std_blue[i] >= first_std_blue):
                count += 1
                buff_one_gist.append(res_std_blue[i])
        if len(buff_one_gist) > 0:
            subs_gist.append(np.mean(buff_one_gist))
            height_gist.append(count)
        first_std_blue += width_gap

    print(subs_gist)
    print(height_gist)
    return subs_gist, height_gist

def init_graphgist_errors(data_directory_1, data_directory_2):
    res_std_blue, res_std_red = device_errors(data_directory_1, data_directory_2)
    subs_gist, height_gist = init_data_std_errors_blue(res_std_blue)
    fig, ax = plt.subplots()
    x = [float(v) for v in subs_gist]
    widths = 0.03
    ax.bar(x, height_gist, width=widths, align='center', color='C0', edgecolor='black', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in x], rotation=45, fontsize=7)
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Count')
    ax.set_title('Histogram (bars at subs_gist)')
    plt.tight_layout()
    plt.show()
    return fig, ax

def init_data_gist_each_point(data_directory_1, data_directory_2):
    data_blue = avaread.read_file(data_directory_1)
    errors_blue = []
    for i in range(len(data_blue)):
        errors_blue.append(data_blue.scope.T[i])

    print(len(errors_blue))

    if data_directory_2 != 0:
        data_red = avaread.read_file(data_directory_2)
        errors_red = []
        for i in range(len(data_red)):
            errors_red.append(data_red.scope.T[i])
        print(len(errors_red))
    else:
        errors_red = 0

    return errors_blue, errors_red

def init_graph_gist_feach(data_directory_1, data_directory_2):
    errors_blue, errors_red = init_data_gist_each_point(data_directory_1, data_directory_2)
    """
        measurements: numpy array shape (n_frames, n_pixels) e.g. (1000, 2047)
        output_dir: папка для сохранения графиков (создаётся если не существует)
        """
    output_dir = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\260518'
    os.makedirs(output_dir, exist_ok=True)

    errors_blue = np.array(errors_blue)
    n_frames, n_pixels = errors_blue.shape
    errors_blue = errors_blue.transpose()
    print(n_frames, n_pixels)

    # ширина и форматируемые параметры можно менять
    for p in range(n_pixels):
        print('Start drawing graph N ' + str(p))
        vals = errors_blue[p]
        print(vals)
        print(len(errors_blue[p]))
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(vals, bins=33, color='C0', edgecolor='black', alpha=0.8)
        ax.set_xlabel('Value', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.set_title(f'Pixel {p}', fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=7)
        plt.tight_layout()
        fname = os.path.join(output_dir, f'hist_pixel_{p:04d}.png')
        fig.savefig(fname)
        plt.close(fig)
        print('Ended graph N ' + str(p))


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

    peak_region_mask = (waves >= 433.4) & (waves <= 434.3)
    x_region = waves[peak_region_mask]
    y_region = chosen_time_spectrum[peak_region_mask]

    A1, A2, A3, A4, sigma_Hg, sigma_Dg, sigma_Hsg, sigma_Dsg, A0 = 0, 3000, 0, 0, 0.1, 0.1, 0.5, 0.5, 0
    #A1, A2, A3, A4, sigma_Hb, sigma_Db, sigma_Hs, sigma_Ds, A0 = 5500, 16650, 0, 0, 0.1, 0.1, 0.5, 0.5, 0
    #A1, A2, sigma_Hb, sigma_Db, A0 = 17000, 42480, 0.1, 0.1, 0

    #попытка построить ошибки
    instrumental_error = [0.05 * i for i in y_region]
    a_sigma = False

    popt_full_1, pcov_full_1, infodict, mesg, ier = curve_fit(
        balmer_Hg_Dg_gauss_4, x_region, y_region,
        p0=[A1, A2, A3, A4, sigma_Hg, sigma_Dg, sigma_Hsg, sigma_Dsg, A0],
        bounds=(0, 1000000),
        full_output=True,
        #sigma=instrumental_error,
        #absolute_sigma=a_sigma
    )

    return waves, chosen_time_spectrum, x_region, y_region, popt_full_1, pcov_full_1, infodict



def init_graph_gauss(chosen_time, data_directory_1, data_directory_2):
    #waves, chosen_time_spectrum, x_region, y_region, popt_full = init_data_gauss(chosen_time, data_directory_1, data_directory_2)
    waves, chosen_time_spectrum, x_region, y_region, popt_full_1, pcov_full_1, infodict = init_data_balmer_Hb_Db_gauss(chosen_time, data_directory_1, data_directory_2)
    # кривая для отображения
    x_fit = np.linspace(x_region.min(), x_region.max(), 500)
    y_fit = balmer_Hg_Dg_gauss_4(x_fit, *popt_full_1)

    print('Невязка\n', infodict['fvec'], '\n')
    perr = np.sqrt(np.diag(pcov_full_1))
    print('Ошибка параметров\n', perr, '\n')

    # Оценка стандартного отклонения остатков
    sigma_estimated = np.std(infodict['fvec'], ddof=len(popt_full_1))

    # Уточнённая ковариационная матрица
    # (исправляем pcov с учётом оценённой погрешности)
    pcov_corrected = pcov_full_1 * sigma_estimated ** 2
    perr_corrected = np.sqrt(np.diag(pcov_corrected))

    print(f"Оценённая погрешность измерений: {sigma_estimated:.4f}")
    print(f"Ошибки параметров с учётом разброса: {perr_corrected}")


    plt.figure(figsize=(10, 6))
    plt.plot(waves, chosen_time_spectrum, 'b-', label='Экспериментальный спектр', alpha=0.7)
    plt.plot(x_region, y_region, 'ro', label='Область аппроксимации', markersize=4)
    plt.plot(x_fit, y_fit, 'r-', label='Результат аппроксимации', linewidth=2)
    # instrumental_error = [0.05 * i for i in y_region]
    # plt.errorbar(x_region, y_region, yerr=instrumental_error, fmt='o', capsize=1, label='Данные с аппаратной погрешностью')

    # info_text = f'A1 = {popt_full_1[0]:.3f} ± {perr_corrected[0]:.3f}\nA2 = {popt_full_1[1]:.3f} ± {perr_corrected[1]:.3f}\nA3 = {popt_full_1[2]:.3f} ± {perr_corrected[2]:.3f} \
    # A4 = {popt_full_1[3]:.3f} ± {perr_corrected[3]:.3f}\nsigma_Hg = {popt_full_1[4]:.3f} ± {perr_corrected[4]:.3f}\nsigma_Dg = {popt_full_1[5]:.3f} ± {perr_corrected[5]:.3f} \
    # sigma_Hsg = {popt_full_1[6]:.3f} ± {perr_corrected[6]:.3f}\nsigma_Dsg = {popt_full_1[7]:.3f} ± {perr_corrected[7]:.3f}\nA0 = {popt_full_1[8]:.3f} ± {perr_corrected[8]:.3f}'
    # plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    #A1, A2, A3, A4, sigma_Hg, sigma_Dg, sigma_Hsg, sigma_Dsg, A0

    info_text = f'A1 = {popt_full_1[0]:.3f} ± {perr[0]:.3f}\nA2 = {popt_full_1[1]:.3f} ± {perr[1]:.3f}\nA3 = {popt_full_1[2]:.3f} ± {perr[2]:.3f} \
        A4 = {popt_full_1[3]:.3f} ± {perr[3]:.3f}\nsigma_Hg = {popt_full_1[4]:.3f} ± {perr[4]:.3f}\nsigma_Dg = {popt_full_1[5]:.3f} ± {perr[5]:.3f} \
        sigma_Hsg = {popt_full_1[6]:.3f} ± {perr[6]:.3f}\nsigma_Dsg = {popt_full_1[7]:.3f} ± {perr[7]:.3f}\nA0 = {popt_full_1[8]:.3f} ± {perr[8]:.3f}'
    plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    y_popt = []
    y_popt.append(popt_full_1[0])
    y_popt.append(popt_full_1[4])
    y_popt.append(popt_full_1[8])

    y_fit_Hg_big = gauss_Hg_big(x_fit, *y_popt)
    plt.plot(x_fit, y_fit_Hg_big, 'r', linestyle='dashed', linewidth=2, label='H_warm')

    y_popt_1 = []
    y_popt_1.append(popt_full_1[1])
    y_popt_1.append(popt_full_1[5])
    y_popt_1.append(popt_full_1[8])

    y_fit_Dg_big = gauss_Dg_big(x_fit, *y_popt_1)
    plt.plot(x_fit, y_fit_Dg_big, 'g', linestyle='dashed', linewidth=2,  label='D_warm')

    y_popt = []
    y_popt.append(popt_full_1[2])
    y_popt.append(popt_full_1[6])
    y_popt.append(popt_full_1[8])
    y_fit_Hg_small = gauss_Hg_small(x_fit, *y_popt)
    plt.plot(x_fit, y_fit_Hg_small, 'r', linestyle='dotted', linewidth=2,  label='H_hot')

    y_popt = []
    y_popt.append(popt_full_1[3])
    y_popt.append(popt_full_1[7])
    y_popt.append(popt_full_1[8])

    y_fit_Dg_small = gauss_Dg_small(x_fit, *y_popt)
    plt.plot(x_fit, y_fit_Dg_small, 'g', linestyle='dotted', linewidth=2,  label='D_hot')

    plt.xlabel('Длина волны (нм)')
    plt.ylabel('Интенсивность')
    plt.legend()
    plt.title('Гауссова аппроксимация спектрального пика')
    plt.grid(True, alpha=0.3)

    plt.show()


def main():
    data_directory_1 = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\260518\Noise blue.STR8'
    #data_directory_1 = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\260518\Noise blue absolute.STR8'
    #data_directory_2 = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\260518\Noise red.STR8'
    #data_directory_2 = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\260518\Noise red absolute.STR8'

    #init_graph(data_directory_1, data_directory_2)

    #init_graph(data_directory_1, data_directory_2=0)
    # chosen_time = 3
    # i_filter = 300
    # init_graph_gauss(chosen_time, data_directory_1, data_directory_2=0)
    #init_graph_autopeaking(chosen_time, i_filter, data_directory_1, data_directory_2=0)
    init_graph_gist_feach(data_directory_1, data_directory_2=0)

main()