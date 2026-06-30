import avaread
import matplotlib.pyplot as plt
import os
import numpy as np
from numba.core.cgutils import sizeof
from numba.cuda.kernels.transpose import transpose

np.set_printoptions(threshold=np.inf)

def fillBkgs(data,count):
    bkgs = []
    for i in range(count):
        bkgs.append(data.scope.T[i])
    return bkgs


def getSpectrum(wave_need, file_path: str, base_width_of_peak, show: bool=False, times: int=15):

    datas = avaread.read_file(file_path)
    print(datas, 'data from file')

    #считывание спектра из одного файла и вычет фона

    bkgs = fillBkgs(datas, 3)
    bkgd = np.average(bkgs, axis=0)
    wave_len = datas.wavelength  # считываем длины волн
    peaks_to_plot, peaks_to_plot_by_shots, peak_one_shot = [], [], []
    for time in range(times):
                final_spectrum = datas.scope.T[time] - bkgd

                for wave_n in range(len(wave_need)):
                    search_wave, base_width_peak = wave_need[wave_n], base_width_of_peak[wave_n]
                    nearest_points = nearest_dot_left_right(wave_len, final_spectrum, search_wave, base_width_peak)
                    print(nearest_points, 'points')
                    if nearest_points:
                        res_peak_area = peak_area(nearest_points, base_width_peak)
                        print(res_peak_area, 'res_peak_area')
                        if not res_peak_area.size > 0:
                            print('0')
                            break
                        peak_one_shot.append(res_peak_area) #в порядке массива wave_need
                        print(peak_one_shot, 'peak_one_shot')

                peaks_to_plot.append(peak_one_shot)
                peak_one_shot = []
                print(peaks_to_plot, 'peaks_to_plot')
    peaks_to_plot_by_shots.append(peaks_to_plot)

    return peaks_to_plot_by_shots


def nearest_dot_left_right (wave_len, final_spectrum, wave_need, base_width_of_peak):
    wave_need_left = min(wave_len, key=lambda x: abs(x - (wave_need - base_width_of_peak/2)))
    wave_need_right = min(wave_len, key=lambda x: abs(x - (wave_need + base_width_of_peak/2)))

    left_dot = np.where(wave_len == wave_need_left)[0][0]
    right_dot = np.where(wave_len == wave_need_right)[0][0]
    if (left_dot == 0) or (right_dot == 0):
        return 0


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

def init_data_continuum_empty(data_directory1):

    #data_directory1 = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\111225\m40 62.STR8'  # файл другого канала

    # wave_need = np.array([584.35995, 551.66, 668.715, 675.2209, 733.813])
    # base_width_of_peak = [0.4, 1., 1.2, 0.7, 1.3]
    wave_need = [464.28, 465.025, 657.8, 658.28, 464.916, 434.74, 435.139, 425.4331, 427.4806, 428.9733,
                 568.125, 434.0625, 485.9375]
    base_width_of_peak = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,]

    peaks_to_plot_by_shots1 = getSpectrum(wave_need, data_directory1, base_width_of_peak, show=True)

    peaks_to_plot_by_shots = np.array(peaks_to_plot_by_shots1)

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

    # con_585 = [584.35995]
    # con_552 = [551.66]
    # con_668 = [668.715]
    # con_674 = [675.2209]
    # con_733 = [733.813]
    #
    # wave_need = []
    # wave_need.append(con_585)
    # wave_need.append(con_552)
    # wave_need.append(con_668)
    # wave_need.append(con_674)
    # wave_need.append(con_733)
    # wave_need.append([])
    return wave_need, peaks_to_plot_by_shots


def init_plots_continuum_empty(data_directory1):

    wave_need, peaks_to_plot_by_shots = init_data_continuum_empty(data_directory1)

    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    fig.suptitle('Possible continuum in diff. p. of the spectrum', fontsize=16)
    name_of_shot = data_directory1.split('\\')[-1]

    print(name_of_shot)
    peaks_to_plot_T = np.transpose(peaks_to_plot_by_shots)

    x_time = [i * 4 for i in range(len(peaks_to_plot_T[0]))]

    # Создаем словарь для удобного доступа к графикам
    plots = {
        'Carbon': axes[0, 0],
        'Oxygen': axes[0, 1],
        'Cromium': axes[0, 2],
        'Nytrogen': axes[1, 0],
        'Deuterium': axes[1, 1],
        'empty': axes[1, 2]  # этот останется пустым
    }

    # fig1, axes1 = plt.subplots(2, 3, figsize=(10, 8))
    # fig1.suptitle('Possible continuum in diff. p. of the spectrum', fontsize=16)
    #
    # plots1 = {
    #     'C': axes1[0, 0],
    #     'O': axes1[0, 1],
    #     'Cr': axes1[0, 2],
    #     'N': axes1[1, 0],
    #     'D': axes1[1, 1],
    #     'empty': axes1[1, 2]  # этот останется пустым
    # }

    name_of_ion = ['II', 'II', 'III', 'III', 'II', 'II', 'I', 'I', 'I', 'II', 'beta', 'gamma']
    grp_nm = ['C', 'C', 'C', 'C', 'O', 'O', 'Cr', 'Cr', 'Cr', 'N', 'D', 'D']

    # cчетчик для peaks_to_plot_T
    peak_index = 0
    # peak_index1 = 0
    # перебираем группы длин волн
    for group_idx, (group_name, wave_group) in enumerate(zip(['Carbon', 'Oxygen', 'Cromium', 'Nytrogen', 'Deuterium', 'empty'], wave_need)):
        if group_name == 'empty' or len(wave_group) == 0:
            continue

        # Получаем соответствующий график
        ax = plots[group_name]


        # Для каждой длины волны в текущей группе
        wave_legend_count = 0
        for wave in wave_group:
            if wave >= 500:
                if peak_index < len(peaks_to_plot_T):
                    ax.plot(x_time, peaks_to_plot_T[peak_index],
                            label=f'{wave} nm ' + grp_nm[wave_legend_count] + ' ' + name_of_ion[wave_legend_count])
                    peak_index += 1
                    wave_legend_count += 1
                else:
                    break

            else:
                if peak_index < len(peaks_to_plot_T):
                    peak_index += 1
                    wave_legend_count += 1
                else:
                    break
            # wave_legend_count += 1

        ax.set_xlabel('Time, (ms)', fontsize=12)
        ax.set_ylabel('Intensity, (a. u.)', fontsize=12)
        ax.set_title(f'{group_name} - Intens. by time', fontsize=12)
        ax.grid(True)
        ax.legend(fontsize=8)

        # не работает!!!!!

        # ax1 = plots1[group_name]
        #
        # for wave in wave_group:
        #     if wave >= 550:
        #         if peak_index1 < len(peaks_to_plot_T):
        #             ax1.plot(x_time, peaks_to_plot_T[peak_index1],
        #                     label=f'{wave} nm')
        #             peak_index1 += 1
        #         else:
        #             break
        #     else:
        #         if peak_index1 < len(peaks_to_plot_T):
        #             continue
        #         else:
        #             break
        #
        #
        # # Настройки графика
        # ax1.set_xlabel('Time, (ms.)', fontsize=12)
        # ax1.set_ylabel('Intensity, (a. u.)', fontsize=12)
        # ax1.set_title(f'{group_name} - Intens. by time, shot # {name_of_shot}', fontsize=12)
        # ax1.grid(True)
        # ax1.legend(fontsize=8)

    # 6-й график пустой
    plots['empty'].set_title(f'Empty - shot # {name_of_shot}', fontsize=12)
    plots['empty'].grid(True)

    # plots1['empty'].set_title(f'Empty - shot # {name_of_shot}', fontsize=12)
    # plots1['empty'].grid(True)

    plt.tight_layout()
    plt.show()



def init_data_continuum_curr_line(selected_wave_len):

    #читаем и вызываем функцию от других агрументов
    impact_param = np.array([79.97, 69.90, 59.80, 49.71, 39.64, 29.61, 19.65, 9.78, 0., -9.65, -19.17, -28.54
                             , -37.75, -46.78, -55.62, -64.27, -72.72])
    lil_radius = np.array([80, 70, 60, 50, 40, 30, 20, 10, 0, -10, -20, -30, -40, -50, -60, -70, -80])

    concat_imp_par_lil_rad = np.array([impact_param, lil_radius])


    data_directory_papka = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\111225'
    files = os.listdir(data_directory_papka)

    print(files)
    cont_to_plot = []
    x_impact_param = []
    base_width_of_peak = [0.1]

    for file in files:
        if int(file[4:6]) >= 60: #Менять для разных длин волн! до 500 <60, от 550 >=60
            print(int(file[4:6]))
            idx: int
            print(int(file[1:3]), file[0:1])
            if (int(file[1:3]) in lil_radius)  and (file[0:1] == 'm'):
                idx = np.argmax(concat_imp_par_lil_rad[1] == int('-'+file[1:3]))
                print('-idx', idx, int('-'+file[1:3]))
                x_impact_param.append(concat_imp_par_lil_rad[0][idx])
            elif (int(file[1:3]) in lil_radius)  and (file[0:1] == 'p'):
                idx = np.argmax(concat_imp_par_lil_rad[1] == int(file[1:3]))
                print('+idx', idx, int('-'+file[1:3]))
                x_impact_param.append(concat_imp_par_lil_rad[0][idx])
            cont_to_plot.append(getSpectrum(selected_wave_len, r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\111225\\'+ file, base_width_of_peak, show=True))

    print(cont_to_plot)
    cont_to_plot = np.array(cont_to_plot)
    print(x_impact_param)
    reshaped_cont_to_plot = []

    for j in range(len(cont_to_plot)):
        flat = cont_to_plot[j].flatten()
        reshaped_cont_to_plot.append(flat)

    reshaped_cont_to_plot = np.array(reshaped_cont_to_plot)
    reshaped_cont_to_plot_T = np.transpose(reshaped_cont_to_plot)

    order, x_impact_param = sort_impact_par(x_impact_param)

    print(order)  # в конце 2 раза одиновая цифра 80, поэтому получается 2 девятки
    sort_resh_cont_to_plot_T = []
    for h in range(len(reshaped_cont_to_plot_T)):
        sort_resh_cont_to_plot_T.append(reshaped_cont_to_plot_T[h][order])

    sort_resh_cont_to_plot_T = np.array(sort_resh_cont_to_plot_T)

    return sort_resh_cont_to_plot_T, x_impact_param

#отсортируем прицельный параметр и конт ту плот в порядке возрастания.
def sort_impact_par(x_impact_param):
    dict_to_sort_im_par = {}
    for l in range(len(x_impact_param)):
        dict_to_sort_im_par[str(x_impact_param[l])] = l
    print(dict_to_sort_im_par)
    x_impact_param = sorted(x_impact_param)
    order = []
    for imp in (x_impact_param):
        order.append(dict_to_sort_im_par[str(imp)])

    return order, x_impact_param


def init_plots_curr_line(selected_wave_len, wave_name):

    sort_resh_cont_to_plot_T, x_impact_param = init_data_continuum_curr_line(selected_wave_len)

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
        '#FFCC00'   # 15 - глубокий желтый (золотой)
    ]

    linestyles = [
        'solid',    # 1
        'dashed',   # 2
        'dotted',   # 3
        'solid',    # 4
        'dashed',   # 5
        'dotted',   # 6
        'solid',    # 7
        'dashed',   # 8
        'dotted',   # 9
        'solid',    # 10
        'dashed',   # 11
        'dotted',   # 12
        'solid',    # 13
        'dashed',   # 14
        'dotted'    # 15
    ]

    x_time = [i * 4 for i in range(15)]

    plt.figure(2)
    plt.title(str(wave_name) + ' ' + str(selected_wave_len))
    for i in range(len(sort_resh_cont_to_plot_T)):
        plt.plot(x_impact_param, sort_resh_cont_to_plot_T[i], color=colors[i], linestyle=linestyles[i],
                 marker='o', markersize=4, label=str(x_time[i]+2)+' ms')

    # for i in range(len(reshaped_cont_to_plot_T)):
    #     plt.plot(x_impact_param, reshaped_cont_to_plot_T[i], label=str(x_time[i]))

    plt.xlabel('r, mm')
    plt.ylabel('Intesity, a. u.')
    plt.grid(True)

    plt.legend()
    plt.show()

def main():

    # data_directory = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\111225\p00 20.STR8'
    # init_plots_continuum_empty(data_directory)
    # data_directory1 = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\111225\p00 22.STR8'
    # init_plots_continuum_empty(data_directory1)
    # data_directory2 = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\111225\p00 33.STR8'
    # init_plots_continuum_empty(data_directory2)
    # data_directory3 = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\111225\p00 35.STR8'
    # init_plots_continuum_empty(data_directory3)
    # data_directory5 = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\111225\p00 36.STR8'
    # init_plots_continuum_empty(data_directory5)
    # data_directory6 = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\111225\p00 37.STR8'
    # init_plots_continuum_empty(data_directory6)
    # data_directory7 = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\111225\p00 38.STR8'
    # init_plots_continuum_empty(data_directory7)
    # data_directory8 = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\111225\p00 39.STR8'
    # init_plots_continuum_empty(data_directory8)
    # data_directory9 = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\111225\p00 40.STR8'
    # init_plots_continuum_empty(data_directory9)
    data_directory10 = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\111225\p00 41.STR8'
    init_plots_continuum_empty(data_directory10)


    # selected_wave_len = [568.125]
    # wave_name = 'N II'
    # init_plots_curr_line(selected_wave_len, wave_name)


main()

