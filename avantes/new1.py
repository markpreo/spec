import avaread
import matplotlib.pyplot as plt
import os
import numpy as np


data_directory = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\111225\p00 41.STR8' #поставить сюда файл одного каналана
data_directory1 = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\111225\p00 64.STR8' # файл другого канала

def getSpectrum(wave_need, file_path: str, show: bool=False):

    datas = avaread.read_file(file_path)
    print(datas, 'data from file')

    n = 0
    bkgs = []

    bkgs.append(datas.scope.T[0])
    bkgs.append(datas.scope.T[1])
    bkgs.append(datas.scope.T[2])
    n += 3

    bkgd = np.average(bkgs, axis=0)
    errors = np.std(bkgs, axis=0)

    bkgd_errors = errors / n**0.5
    #считывание спектра из одного файла и вычет фона

    np.set_printoptions(threshold=np.inf)
    raw_spectrum = np.zeros_like(datas.scope.T[0])

    final_spectrum = []
    time_from_data = 15
    base_width_of_peak = 0.1
    peaks_to_plot_by_shots = []

    peaks_time_order = {}
    peaks_to_plot = []
    wave_len = datas.wavelength  # считываем длины волн
    for time in range(time_from_data):
                raw_spectrum = datas.scope.T[time]  # считывание спектров
                final_spectrum = raw_spectrum - bkgd
                peaks_time_order[time] = {}
                peak_one_shot = []
                for wave_n in wave_need:
                    peaks_time_order[time][wave_n] = {}
                    res_nearest_dot_left_right = nearest_dot_left_right(wave_len, final_spectrum, wave_n, base_width_of_peak)
                    peaks_time_order[time][wave_n] = res_nearest_dot_left_right
                    res_peak_area = peak_area(res_nearest_dot_left_right, base_width_of_peak)
                    peaks_time_order[time][wave_n].append(res_peak_area)
                    peak_one_shot.append(res_peak_area) #в порядке массива wave_need



                peaks_to_plot.append(peak_one_shot)
    peaks_to_plot_by_shots.append(peaks_to_plot)

    return peaks_to_plot_by_shots


def nearest_dot_left_right (wave_len, final_spectrum, wave_need, base_width_of_peak):
    wave_need_left = min(wave_len, key=lambda x: abs(x - (wave_need - base_width_of_peak/2)))
    wave_need_right = min(wave_len, key=lambda x: abs(x - (wave_need + base_width_of_peak/2)))

    left_dot = np.where(wave_len == wave_need_left)[0][0]
    right_dot = np.where(wave_len == wave_need_right)[0][0]

    res_intense_and_edge_points = [final_spectrum[left_dot:right_dot+1], wave_len[left_dot-1:right_dot+1]]
    return  res_intense_and_edge_points #возвращает значения интенсивностей между точками на небольшом расстоянии (либо 1 значение если точки совпадают)
    # сами длины волн и это расстояние


def peak_area (res_inte_ed_p, base_width_of_peak):
    res_peak_area = 0
    area_width_by_colomn = []
    for j in range(1, len(res_inte_ed_p[1])):
        area_width_by_colomn.append(res_inte_ed_p[1][j] - res_inte_ed_p[1][j-1])



    if len(res_inte_ed_p[0]) == len(area_width_by_colomn):
        for i in range(len(res_inte_ed_p[0])):
            res_peak_area += area_width_by_colomn[i] * res_inte_ed_p[0][i]
            #print('res_peak_area += ', area_width_by_colomn[i], '*', res_inte_ed_p[0][i], '==', res_peak_area)


    if res_peak_area == 0:
        res_peak_area = base_width_of_peak * res_inte_ed_p[0][0] #если точка одна, умножаем базовую ширину пика на ее интенсивность
        # print('res_peak_area += ', base_width_of_peak, '*', res_inte_ed_p[0][0], '==', res_peak_area)


    return abs(res_peak_area)


wave_need = [464.28, 465.025, 657.8, 658.28, 464.916, 434.74, 435.139, 425.4331, 427.4806, 428.9733,
            568.125, 434.0625, 485.9375]

peaks_to_plot_by_shots2 = getSpectrum(wave_need, data_directory, show=True)
peaks_to_plot_by_shots1 = getSpectrum(wave_need, data_directory1, show=True)

peaks_to_plot_by_shots = np.array(peaks_to_plot_by_shots1) + np.array(peaks_to_plot_by_shots2)

wave_label_C = [464.28, 465.025, 657.8, 658.28]
wave_label_O = [464.916, 434.74, 435.139]
wave_label_Cr = [425.4331, 427.4806, 428.9733]
wave_label_N = [568.125]
wave_label_D = [434.0625, 485.9375]

wave_need = []
wave_need.append(wave_label_C)
wave_need.append(wave_label_O)
wave_need.append(wave_label_Cr)
wave_need.append(wave_label_N)
wave_need.append(wave_label_D)
wave_need.append([])

fig, axes = plt.subplots(2, 3, figsize=(10, 8))

name_of_shot = data_directory.split('\\')[-1] + ' ' + data_directory1.split('\\')[-1]

print(name_of_shot)
peaks_to_plot_T = np.transpose(peaks_to_plot_by_shots)

x_time = [i * 4 for i in range(len(peaks_to_plot_T[0]))]

# Создаем словарь для удобного доступа к графикам
plots = {
    'C': axes[0, 0],
    'O': axes[0, 1],
    'Cr': axes[0, 2],
    'N': axes[1, 0],
    'D': axes[1, 1],
    'empty': axes[1, 2]  # этот останется пустым
}

# cчетчик для peaks_to_plot_T
peak_index = 0

# перебираем группы длин волн
for group_idx, (group_name, wave_group) in enumerate(zip(['C', 'O', 'Cr', 'N', 'D', 'empty'], wave_need)):
    if group_name == 'empty' or len(wave_group) == 0:
        continue

    # Получаем соответствующий график
    ax = plots[group_name]

    # Для каждой длины волны в текущей группе
    for wave in wave_group:
        if peak_index < len(peaks_to_plot_T):
            ax.plot(x_time, peaks_to_plot_T[peak_index],
                    label=f'{wave} nm')
            peak_index += 1
        else:
            break


    # Настройки графика
    ax.set_xlabel('Time, (ms.)', fontsize=12)
    ax.set_ylabel('Intensity, (rel. u.)', fontsize=12)
    ax.set_title(f'{group_name} - Intens. by time, shot # {name_of_shot}', fontsize=12)
    ax.grid(True)
    ax.legend(fontsize=8)

# 6-й график пустой
plots['empty'].set_title(f'Empty - shot # {name_of_shot}', fontsize=12)
plots['empty'].grid(True)

plt.tight_layout()
plt.show()