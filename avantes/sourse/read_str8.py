import avaread
import matplotlib.pyplot as plt
import os
import numpy as np
from getLinesFromNISTCSV import getObservedLinesNIST

blue_shot_numbers = (20, 56)
data_directory = r'111225'
time2plot = 8
blue_shots2average = (6, 15)

red_shot_numbers = (60, 71)
red_shots2average = (6, 6)


def getSpectrum(directory: str, shot_numbers: tuple, shots2average: tuple, time2plot: int, show: bool=False):
    files = os.listdir(directory)
    files = [file for file in files if os.path.splitext(file)[-1] == '.STR8'
             and os.path.splitext(file)[0].split(' ')[1].isdigit()
             and shot_numbers[0] <= int(os.path.splitext(file)[0].split(' ')[1]) <= shot_numbers[1]]

    #print('Files loaded:\n', files)
    #в datas информация о файлах
    datas = [avaread.read_file(os.path.join(directory, file)) for file in files]

    #print(datas[0].scope.T[0]) фон конкретного времени
    #расчет поля среднего
    n = 0
    bkgs = []
    for data in datas:
        # bkgd += np.average(data.scope.T[0:3], axis=0)
        bkgs.append(data.scope.T[0])
        bkgs.append(data.scope.T[1])
        bkgs.append(data.scope.T[2])
        n += 3

    bkgd = np.average(bkgs, axis=0)
    #print(bkgd)
    errors = np.std(bkgs, axis=0)

    bkgd_errors = errors / n**0.5
    #считывание спектра из одного файла и вычет фона

    np.set_printoptions(threshold=np.inf)
    raw_spectrum = np.zeros_like(datas[0].scope.T[0])
    m = 0
    final_spectrum = []

    base_width_of_peak = 0.1
    peaks_to_plot = []
    # peaks = []
    # peaks_order = {}
    peaks_time_order = {}
    wave_need = [464.28, 465.025, 657.8, 658.28, 464.916, 434.74, 435.139, 425.4331, 427.4806, 428.9733,
                 568.125, 434.0625, 485.9375]

    peaks_to_plot_by_shots = []


    for i in range(shots2average[0], shots2average[1]+1):
        peaks_time_order['shots num '+ str(i)] = {}
        peaks_to_plot = []
        print(i)
        print('время из даты', len(datas[i].scope.T))
        for time in range(15):
            raw_spectrum = datas[i].scope.T[time]  # считывание спектров
            wave_len = datas[i].wavelength #считываем длины волн
            final_spectrum = raw_spectrum - bkgd
            peaks_time_order['shots num '+ str(i)][time] = {}
            #print("time", time)
            peak_one_shot = []
            for wave_n in wave_need:
                #print("wave", wave_n)
                #print(nearest_dot(wave_len, final_spectrum, wave_n))
                peaks_time_order['shots num '+ str(i)][time][wave_n] = {}
                # print(nearest_dot_left_right(wave_len, final_spectrum, wave_n, base_width_of_peak))
                res_nearest_dot_left_right = nearest_dot_left_right(wave_len, final_spectrum, wave_n, base_width_of_peak)
                peaks_time_order['shots num '+ str(i)][time][wave_n] = res_nearest_dot_left_right
                res_peak_area = peak_area(res_nearest_dot_left_right, base_width_of_peak)
                peaks_time_order['shots num ' + str(i)][time][wave_n].append(res_peak_area)
                if shots2average[0] != shots2average[1]:
                    peak_one_shot.append(res_peak_area) #в порядке массива wave_need
                # print(peaks_time_order)

            # peaks_time_order[time] = peaks_order # внешний словарь с ключами по времени
            m += 1
            peaks_to_plot.append(peak_one_shot)
        peaks_to_plot_by_shots.append(peaks_to_plot)

    # try:
    #     print(peaks_time_order)
    # except:
    #     print(shots2average)

    # print(peaks_to_plot)
    raw_spectrum /= m
    raw_spectrum_errors = errors / m**0.5
    #print(raw_spectrum)
    final_spectrum = raw_spectrum - bkgd
    final_errors = np.sqrt(np.square(bkgd_errors) + np.square(raw_spectrum_errors))

    if show:
        plt.plot(datas[0].wavelength, bkgd, 'b-')
        plt.fill_between(
             datas[0].wavelength,
             bkgd + bkgd_errors,
             bkgd - bkgd_errors,
             color="b",
             alpha=0.12,
             zorder=0)

        plt.plot(datas[8].wavelength, raw_spectrum, 'r-')
        plt.fill_between(
             datas[0].wavelength,
             raw_spectrum + raw_spectrum_errors,
             raw_spectrum - raw_spectrum_errors,
             color="r",
             alpha=0.12,
             zorder=0)

        plt.plot(datas[8].wavelength, final_spectrum, 'go-')
        plt.plot(final_spectrum, 'go-')
        plt.fill_between(
            datas[0].wavelength,
            final_spectrum + final_errors,
            final_spectrum - final_errors,
            color="g",
            alpha=0.12,
            zorder=0)

        plt.axhline(0, color='k')
        plt.grid()
        plt.xlabel('Lambda, nm')
        plt.ylabel('Intensity, ADC counts')

        # plt.show()
        return peaks_to_plot_by_shots


def plotLinesFromDict(lines: dict, color: str, label: str):
    vals = []
    for key in lines.keys():
        vals.append(lines[key])
    plt.vlines(vals, 0, 65536-1, color=color, label=label)
    plt.legend()


def plotLinesFromList(lines: list, color: str, label: str, linestyles: str):
    plt.vlines(lines, 0, 65536-1, color=color, label=label, linestyles=linestyles)
    plt.legend()


def plotLinesFromNISTDict(dictionary, color: str):
    for stage in dictionary.keys():
        lines = []
        for line in dictionary[stage]:
            if line['intensity'] >= 0:
                lines.append(line['wl'])
                # text = f'{stage} {line['wl']} ({line['intensity']})'
                # plt.text(line['wl'], -100, text, fontsize='small', rotation='vertical')
        stageNumber = dictionary[stage][0]['stageNumber']
        if stageNumber == 1:
            style = 'solid'
        elif stageNumber == 2:
            style = 'dashed'
        elif stageNumber == 3:
            style = 'dashdot'
        else:
            style = 'dotted'
        plt.vlines(lines, 0, 65536 - 1, label=stage, color=color, linestyles=style)
        # plt.vlines(lines, -100, 0, color='k')
    plt.legend()


def plot_bar(C_bar_x, C_bar_y, colors):
    plt.bar(C_bar_x, C_bar_y, align='center', alpha=1, color=colors)


def nearest_dot_left_right (wave_len, final_spectrum, wave_need, base_width_of_peak):
    # base_width_of_peak = 0.1 #предполагаемая ширина пика
    wave_need_left = min(wave_len, key=lambda x: abs(x - (wave_need - base_width_of_peak/2)))
    wave_need_right = min(wave_len, key=lambda x: abs(x - (wave_need + base_width_of_peak/2)))

    left_dot = np.where(wave_len == wave_need_left)[0][0]
    right_dot = np.where(wave_len == wave_need_right)[0][0]

    # print(final_spectrum[left_dot])
    # print(final_spectrum[right_dot])
    res_intense_and_edge_points = [final_spectrum[left_dot:right_dot+1], wave_len[left_dot-1:right_dot+1]]
    return  res_intense_and_edge_points #возвращает значения интенсивностей между точками на небольшом расстоянии (либо 1 значение если точки совпадают)
    # сами длины волн и это расстояние


def peak_area (res_inte_ed_p, base_width_of_peak):
    res_peak_area = 0
    area_width_by_colomn = []
    for j in range(1, len(res_inte_ed_p[1])):
        area_width_by_colomn.append(res_inte_ed_p[1][j] - res_inte_ed_p[1][j-1])

    #print(area_width_by_colomn, res_inte_ed_p[0])
    if len(res_inte_ed_p[0]) == len(area_width_by_colomn):
        for i in range(len(res_inte_ed_p[0])):
            res_peak_area += area_width_by_colomn[i] * res_inte_ed_p[0][i]
        # print('щирина', res_peak_area)
    if res_peak_area == 0:
        res_peak_area = base_width_of_peak * res_inte_ed_p[0] #если точка одна, умножаем базовую ширину пика на ее интенсивность
    return res_peak_area

peaks_to_plot_by_shots = getSpectrum(data_directory, blue_shot_numbers, blue_shots2average, time2plot, show=True)
getSpectrum(data_directory, red_shot_numbers, red_shots2average, time2plot, show=True)
peaks_to_plot_by_shots = np.array(peaks_to_plot_by_shots)

# print(peaks_to_plot_by_shots)
# print('длина внеш массива', len(peaks_to_plot_by_shots))
# print('длина сред массива', len(peaks_to_plot_by_shots[0]))
# print('длина внут массива', len(peaks_to_plot_by_shots[0][0]))

#wave_need = [464.28, 464.916, 465.025, 434.74, 435.139, 657.8, 658.28, 425.4331, 427.4806, 428.9733,
#                 568.125, 434.0625, 485.9375] #тупо продублирован из функции!!!
#ions_label_to_plot = ['C III', 'O II', 'C III', 'O II', 'O II', 'C II', 'C II', 'Cr I', 'Cr I', 'Cr I',
#                      'N II', 'Dgamma', 'Dbeta']
# wave_need = [464.28, 465.025, 657.8, 658.28, 464.916, 434.74, 435.139, 425.4331, 427.4806, 428.9733,
#              568.125, 434.0625, 485.9375]


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


# for l in range(len(peaks_to_plot_by_shots)):
#     peaks_to_plot_T = np.transpose(peaks_to_plot_by_shots[l])
#     # print(peaks_to_plot_T)
#     # print('длина сред массива', len(peaks_to_plot_T))
#     x_time = [i * 4 for i in range(len(peaks_to_plot_T[0]))]


num_of_shot = 6
peaks_to_plot_T = np.transpose(peaks_to_plot_by_shots[num_of_shot])
#plt.figure('величина пиков от времени для шот №' + str(num_of_shot))
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

num_of_shot = 6
peaks_to_plot_T = np.transpose(peaks_to_plot_by_shots[num_of_shot])
x_time = [i * 4 for i in range(len(peaks_to_plot_T[0]))]

# Счетчик для peaks_to_plot_T
peak_index = 0

# Перебираем группы длин волн
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
    ax.set_ylabel('Intensity, (отн. ед.)', fontsize=12)
    ax.set_title(f'{group_name} - Intensity by time, shot # {num_of_shot}', fontsize=14)
    ax.grid(True)
    ax.legend(fontsize=8)

# 6-й график (пустой) можно оставить без изменений или добавить текст
plots['empty'].set_title(f'Empty - shot # {num_of_shot}', fontsize=14)
plots['empty'].grid(True)

plt.tight_layout()
plt.show()

# for k in range(len(peaks_to_plot_T)):
#     # print('длина внут массива', len(peaks_to_plot_T[k]))
#     #plt.figure(str(k) + ' величина пика №k от времени')
#     plt.plot(x_time, peaks_to_plot_T[k], label=str(ions_label_to_plot[k]) + ' ' + str(wave_need[k]))
#     plt.xlabel('Time, (ms.)', fontsize=12)
#     plt.ylabel('Intensity, (отн. ед.)', fontsize=12)
#     plt.title('Intensity by time, shot # ' + str(num_of_shot), fontsize=14)
#     plt.grid(True)
#     plt.legend()



plt.show()

# lines_H = {'Halpha': 656.278,
#          'Hbeta': 486.127,
#          'Hgamma': 434.040,
#          'Hdelta': 410.166}

lines_HI = np.array([6562.8518, 6562.7248, 6562.7110, 4861.3615, 4861.2870, 4861.2786, 4340.462, 4101.74]) / 10
lines_CI = np.array([6013.22, 5380.34]) / 10
lines_CII = np.array([7236.42, 7231.32, 6582.88, 6578.05, 5889.77, 5151.09, 5145.16, 4267.258, 4267.003]) / 10
lines_NI = np.array([7468.31, 7442.29, 7423.64, 6482.70, 5752.50, 4963.98]) / 10
lines_NII = np.array([6610.56, 6482.05, 5941.65, 5931.78, 5710.77, 5686.21, 5679.56, 5676.02, 5666.63, 5045.10, 5010.62, 5007.32, 5005.15, 5001.48, 4994.36, 4803.29, 4643.08, 4630.54, 4621.39, 4607.16, 4601.48, 4447.03, 4241.7]) / 10
lines_SI = np.array([6757.16, 4694.13]) / 10
lines_SII = np.array([7629.740, 7578.909, 5659.985, 5639.972, 5606.151, 5509.718, 5473.620, 5453.828, 5432.815, 5428.667, 4162.665, 4153.064]) / 10
lines_HeI = np.array([7281.35, 7065.7086, 7065.2153, 7065.1771, 6678.1517, 5875.9663, 5875.6404, 5875.6148, 5047.74, 5015.678, 4921.931, 4713.38, 4713.146, 4471.68, 4471.479, 4437.55, 4387.929, 4143.76, 4120.99, 4120.82]) / 10
lines_HeII = np.array([6560.10, 5411.52, 4685.8041, 4685.7044, 4685.7038, 4685.4072, 4685.3769]) / 10
lines_OI = np.array([7254.45, 7254.15, 7002.23, 6455.98, 6158.18, 6156.77, 6155.98]) / 10
lines_OII = np.array([4705.352, 4676.235, 4661.633, 4649.135, 4641.810, 4596.175, 4590.972, 4416.974, 4414.905, 4349.426, 4317.138, 4189.789, 4185.449, 4119.215, 4075.862, 4072.157]) / 10
lines_FeI = np.array([5397.1279, 5371.4897, 5341.0239, 5328.5317, 5328.0386, 5270.3564, 5269.5376, 5227.1509, 5171.5962, 5167.4883, 4957.5967, 4920.5029, 4461.6528, 4427.2979, 4415.1226, 4404.7505, 4383.5449, 4375.9302, 4325.7622, 4307.9023, 4282.4028, 4271.7607, 4260.4746, 4250.7871, 4216.1836, 4202.0293, 4143.8682, 4132.0581, 4071.7380, ]) / 10
lines_FeII = np.array([6456.38, 6247.56]) / 10
lines_NiII = np.array([4992.024]) / 10
lines_CuI = np.array([5782.13, 5700.24, 5292.52, 5218.20, 5153.24, 5105.54]) / 10
lines_CuII = np.array([7664.648, 7652.333, 7404.354, 6641.396, 6624.292, 6481.437, 6470.168, 6448.559, 6423.884, 6377.840, 6301.009, 6273.349, 6219.844, 6216.939, 6154.222, 6150.384, 6000.120, 5051.793, 4953.724, 4931.698, 4909.734, 4651.12]) / 10
lines_CrI = np.array([7462.35, 7400.22, 5791.00, 5409.78, 5348.30, 5345.77, 5328.36, 5298.29, 5296.69, 5265.73, 5264.16, 5247.58, 5208.415, 5206.021, 5204.505, 4922.276, 4887.013, 4870.79, 4789.324, 4756.09, 4737.33, 4718.43, 4708.02, 4698.46, 4652.155, 4651.285, 4646.151, 4626.181, 4616.120, 4613.36, 4600.745, 4591.405, 4580.045, 4545.946, 4544.607, 4540.715, 4540.50, 4535.714, 4530.72, 4526.458, 4496.85, 4384.974, 4371.279, 4359.647, 4351.77, 4351.055, 4344.51, 4339.74, 4339.45, 4337.566, 4289.733, 4274.806, 4254.331, 4179.27, 4174.808, 4163.627, 4126.513]) / 10


#plotLinesFromDict(lines_HI, color='orange', label='HI')
# plotLinesFromList(lines_HI, color='orange', linestyles='solid', label='HI')
# plotLinesFromList(lines_CI, color='darkblue', linestyles='solid', label='CI')
# plotLinesFromList(lines_CII, color='darkblue', linestyles='dashed', label='CII')
# plotLinesFromList(lines_NI, color='darkred', linestyles='solid', label='NI')
# plotLinesFromList(lines_NII, color='darkred', linestyles='dashed', label='NII')
# plotLinesFromList(lines_SI, color='yellow', linestyles='solid', label='SI')
# plotLinesFromList(lines_SII, color='yellow', linestyles='dashed', label='SII')
# plotLinesFromList(lines_HeI, color='darkgreen', linestyles='solid', label='HeI')
# plotLinesFromList(lines_HeII, color='darkgreen', linestyles='dashed', label='HeII')
# plotLinesFromList(lines_OI, color='fuchsia', linestyles='solid', label='OI')
# plotLinesFromList(lines_OII, color='fuchsia', linestyles='dashed', label='OII')
# plotLinesFromList(lines_FeI, color='lime', linestyles='solid', label='FeI')
# plotLinesFromList(lines_FeII, color='lime', linestyles='dashed', label='FeII')
# plotLinesFromList(lines_NiII, color='aqua', linestyles='dashed', label='NiII')
# plotLinesFromList(lines_CuI, color='peru', linestyles='solid', label='CuI')
# plotLinesFromList(lines_CuII, color='peru', linestyles='dashed', label='CuII')
# plotLinesFromList(lines_CrI, color='slategray', linestyles='solid', label='CrI')

lines_C = getObservedLinesNIST('C_200_2000.txt')
#plotLinesFromNISTDict(lines_C, color='C0')

# lines_O = getObservedLinesNIST('O_200_2000.txt')
# plotLinesFromNISTDict(lines_O, color='C1')

# lines_N = getObservedLinesNIST('N_200_2000.txt')
# plotLinesFromNISTDict(lines_N, color='C3')

# lines_Fe = getObservedLinesNIST('Fe_200_2000.txt')
# plotLinesFromNISTDict(lines_Fe, color='C4')

# lines_Ni = getObservedLinesNIST('Ni_200_2000.txt')
# plotLinesFromNISTDict(lines_Ni, color='C4')

lines_Cr = getObservedLinesNIST('Cr_200_2000.txt')
#plotLinesFromNISTDict(lines_Cr, color='C4')

# lines_Cu = getObservedLinesNIST('Cu_200_2000.txt')
# plotLinesFromNISTDict(lines_Cu, color='C4')

# lines_S = getObservedLinesNIST('S_200_2000.txt')
# plotLinesFromNISTDict(lines_S, color='C4')

# lines_Mn = getObservedLinesNIST('Mn_200_2000.txt')
# plotLinesFromNISTDict(lines_Mn, color='C4')

# lines_Ag = getObservedLinesNIST('Ag_200_2000.txt')
# plotLinesFromNISTDict(lines_Ag, color='C4')

# lines_Ar = getObservedLinesNIST('Ar_200_2000.txt')
# plotLinesFromNISTDict(lines_Ar, color='C4')

#lines_He = getObservedLinesNIST('He_200_2000.txt')
#plotLinesFromNISTDict(lines_He, color='C5')

#lines_H = getObservedLinesNIST('H_200_2000.txt')
#plotLinesFromNISTDict(lines_H, color='C6')

#данные диаграмм
CI_bar_y = []
CI_bar_x = []
for i in range(len(lines_C['CI'])):
    CI_bar_y.append(lines_C['CI'][i]['intensity'])
    CI_bar_x.append(lines_C['CI'][i]['wl'])

CII_bar_y = []
CII_bar_x = []
for i in range(len(lines_C['CII'])):
    CII_bar_y.append(lines_C['CII'][i]['intensity'])
    CII_bar_x.append(lines_C['CII'][i]['wl'])

CIII_bar_y = []
CIII_bar_x = []
for i in range(len(lines_C['CIII'])):
    CIII_bar_y.append(lines_C['CIII'][i]['intensity'])
    CIII_bar_x.append(lines_C['CIII'][i]['wl'])

C_bar_y = CI_bar_y + CII_bar_y + CIII_bar_y
C_bar_x = CI_bar_x + CII_bar_x + CIII_bar_x
#print(CI_bar_y)
#print(CI_bar_x)

#C_bar_x = [464.74]
#C_bar_y = [5711]
for i in range(len(C_bar_y)):
    C_bar_y[i] = C_bar_y[i] * 9.518


#print(len(C_bar_x), len(C_bar_y))

colors='orange'

#plot_bar(C_bar_x, C_bar_y, colors)

#for i, x in enumerate(C_bar_x):
    #plt.text(i, x, str(x), ha='center')

#plt.xticks(C_bar_x, C_bar_y)
#plt.legend()

#plt.xlim((408, 771))
#plt.show()
