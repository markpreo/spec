import avaread
import matplotlib.pyplot as plt
import os
import numpy as np
from numba.core.cgutils import sizeof
from numba.cuda.kernels.transpose import transpose

np.set_printoptions(threshold=np.inf)

def fillBkgs(datas, low_signal_times):
    bkgs = []
    for i in range(low_signal_times):
        bkgs.append(datas.scope.T[i])
    return bkgs


def getSpectrum(wave_need, file_path: str, base_width_of_peak, show: bool=False, times: int=15):

    datas = avaread.read_file(file_path)
    print(datas, 'data from file')
    low_signal_times = 3

    #считаем фон

    bkgs = fillBkgs(datas, low_signal_times)
    bkgd = np.average(bkgs, axis=0)
    wave_len = datas.wavelength  # считываем длины волн
    peaks_to_plot, peaks_to_plot_by_time, peaks_to_plot_one_time = [], [], []

    # оргвнизовываем считанные пики с формате
    # если файл один, то получаем что peak_one_shot лишнее
    # [[[peak_one_shot], [peak_one_shot], [peak_one_shot]],
    # [[peak_one_shot], [peak_one_shot], [peak_one_shot]],
    # [[peak_one_shot], [peak_one_shot], [peak_one_shot]]]

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
                        peaks_to_plot_one_time.append(res_peak_area)#в порядке массива wave_need
                    else:
                        peaks_to_plot_one_time.append(0)
                peaks_to_plot.append(peaks_to_plot_one_time)
                peaks_to_plot_one_time = []
                print(peaks_to_plot, 'peaks_to_plot')
    peaks_to_plot_by_time.append(peaks_to_plot)

    return peaks_to_plot_by_time


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

def init_data_continuum_empty(data_directory1, data_directory2 = 0):

    #data_directory1 = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\111225\m40 62.STR8'  # файл другого канала

    # wave_need = np.array([584.35995, 551.66, 668.715, 675.2209, 733.813])
    # base_width_of_peak = [0.4, 1., 1.2, 0.7, 1.3]
    wave_need = [464.28, 465.025, 657.8, 658.28, 464.916, 434.74, 435.139, 425.4331, 427.4806, 428.9733,
                 568.125, 434.0625, 485.9375, 551.66, 584.35995, 675.2209]
    base_width_of_peak = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1., 0.4, 0.7]

    peaks_to_plot_by_shots1 = getSpectrum(wave_need, data_directory1, base_width_of_peak, show=True)
    peaks_to_plot_by_shots = list

    if data_directory2 != 0:
        peaks_to_plot_by_shots2 = getSpectrum(wave_need, data_directory2, base_width_of_peak, show=True)

        peaks_to_plot_by_shots = np.array(peaks_to_plot_by_shots1) + np.array(peaks_to_plot_by_shots2)
    else:
        peaks_to_plot_by_shots = np.array(peaks_to_plot_by_shots1)

    wave_label_C = [464.28, 465.025, 657.8, 658.28]
    wave_label_O = [464.916, 434.74, 435.139]
    wave_label_Cr = [425.4331, 427.4806, 428.9733]
    wave_label_N = [568.125]
    wave_label_D = [434.0625, 485.9375]
    wave_label_cont = [551.66, 584.35995, 675.2209]

    wave_need = []
    wave_need.append(wave_label_C)
    wave_need.append(wave_label_O)
    wave_need.append(wave_label_Cr)
    wave_need.append(wave_label_N)
    wave_need.append(wave_label_D)
    wave_need.append(wave_label_cont)

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


def init_plots_continuum_empty(data_directory1, data_directory2 = 0):

    wave_need, peaks_to_plot_by_shots = init_data_continuum_empty(data_directory1, data_directory2)

    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    fig.suptitle('Lines intensity', fontsize=16)
    if data_directory2 != 0:
        name_of_shot = data_directory1.split('\\')[-1] + ' and ' + data_directory2.split('\\')[-1]
    else:
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
        'Continuum': axes[1, 2]  # этот останется пустым
    }

    name_of_ion = ['II', 'II', 'III', 'III', 'II', 'II', 'II', 'I', 'I', 'I', 'II', '$\\gamma$', '$\\beta$', '', '', '']
    grp_nm = ['C', 'C', 'C', 'C', 'O', 'O', 'O', 'Cr', 'Cr', 'Cr', 'N', 'D', 'D', '', '', '']

    # cчетчик для peaks_to_plot_T
    peak_index = 0
    # peak_index1 = 0
    # перебираем группы длин волн
    wave_legend_count = 0
    for group_idx, (group_name, wave_group) in enumerate(zip(['Carbon', 'Oxygen', 'Cromium', 'Nytrogen', 'Deuterium', 'Continuum'], wave_need)):
        # if group_name == 'empty' or len(wave_group) == 0:
        #     continue

        # Получаем соответствующий график
        ax = plots[group_name]



        # Для каждой длины волны в текущей группе

        for wave in wave_group:
                if peak_index < len(peaks_to_plot_T):
                    ax.plot(x_time, peaks_to_plot_T[peak_index],
                            label=grp_nm[wave_legend_count] + ' ' + name_of_ion[wave_legend_count] + ' ' + f'{wave} nm ')
                    peak_index += 1
                    print(wave_legend_count)
                else:
                    break
                wave_legend_count += 1



        ax.set_xlabel('Time, (ms)', fontsize=12)
        ax.set_ylabel('Intensity, (a. u.)', fontsize=12)
        ax.set_title(f'{group_name}', fontsize=12)
        ax.grid(True)
        ax.legend(fontsize=8)


    # 6-й график пустой

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
        if int(file[4:6]) < 60: #Менять для разных длин волн! до 500 <60, от 550 >=60
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
    plt.title('Линия ' + wave_name + ' ' + str(selected_wave_len[0]) + ' nm')
    for i in range(len(sort_resh_cont_to_plot_T)):
        plt.plot(x_impact_param, sort_resh_cont_to_plot_T[i], color=colors[i], linestyle=linestyles[i], label=str(x_time[i]+2))

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
    data_directory10 = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\111225\m60 54.STR8'
    data_directory11 = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\111225\m60 61.STR8'
    init_plots_continuum_empty(data_directory10, data_directory11)


    selected_wave_len = [568.125]
    wave_name = 'N II'
    init_plots_curr_line(selected_wave_len, wave_name)


main()

