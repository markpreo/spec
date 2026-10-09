import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from functools import partial
from scipy.integrate import quad
import avaread
import matplotlib.patches as patches
import math
import os
import json
from lmfit import Model
from after_shot_analysys import calc_peaks_area_wo_integ



def init_data_by_impact_param(data_directory_folder, curr_waves, gaps):

    impact_param = np.array([79.97, 69.90, 59.80, 49.71, 39.64, 29.61, 19.65, 9.78, 0., -9.65, -19.17, -28.54
                             , -37.75, -46.78, -55.62, -64.27, -72.72])
    lil_radius = np.array([80, 70, 60, 50, 40, 30, 20, 10, 0, -10, -20, -30, -40, -50, -60, -70, -80])

    concat_imp_par_lil_rad = np.array([impact_param, lil_radius])
    files = os.listdir(data_directory_folder)

    profiles_to_plot = []
    x_impact_param = []

    for file in files:
        if int(file[4:6]) < 48: #Менять для разных длин волн! до 500 <?, от 550 >=?
            if (int(file[1:3]) in lil_radius)  and (file[0:1] == 'm'):
                idx = np.argmax(concat_imp_par_lil_rad[1] == int('-'+file[1:3]))
                print('-idx', idx, int('-'+file[1:3]))
                x_impact_param.append(concat_imp_par_lil_rad[0][idx])
            elif (int(file[1:3]) in lil_radius)  and (file[0:1] == 'p'):
                idx = np.argmax(concat_imp_par_lil_rad[1] == int(file[1:3]))
                print('+idx', idx, int('-'+file[1:3]))
                x_impact_param.append(concat_imp_par_lil_rad[0][idx])
            peaks_area_wo_integ = calc_peaks_area_wo_integ(str(data_directory_folder) + file, curr_waves, gaps)
            profiles_to_plot.append(peaks_area_wo_integ)


    profiles_to_plot = np.array(profiles_to_plot)
    profiles_to_plot_T = np.transpose(profiles_to_plot)
    reshaped_profiles_to_plot = []
    reshaped_profiles_to_plot_T = profiles_to_plot_T[0]

    order, x_impact_param = sort_impact_par(x_impact_param)

    sort_resh_profiles_to_plot_T = []
    for i in range(len(reshaped_profiles_to_plot_T)):
        sort_resh_profiles_to_plot_T.append([])
        for h in range(len(reshaped_profiles_to_plot_T[i])):
            sort_resh_profiles_to_plot_T[i].append(reshaped_profiles_to_plot_T[i][h][order])

    sort_resh_profiles_to_plot_T = np.array(sort_resh_profiles_to_plot_T)

    return sort_resh_profiles_to_plot_T, x_impact_param

#отсортируем прицельный параметр и конт ту плот в порядке возрастания.
def sort_impact_par(x_impact_param):
    dict_to_sort_im_par = {}
    for l in range(len(x_impact_param)):
        dict_to_sort_im_par[str(x_impact_param[l])] = l

    x_impact_param = sorted(x_impact_param)
    order = []
    for imp in (x_impact_param):
        order.append(dict_to_sort_im_par[str(imp)])

    return order, x_impact_param



def save_data_to_json_imp(sort_resh_profiles_to_plot_T, x_impact_param, data_directory_folder, waves, gaps):

    file_name = data_directory_folder[-8:-2]
    print(file_name)
    data = {}
    for wave in range(len(waves)):
        profiles_array = []
        for time in range(len(sort_resh_profiles_to_plot_T[0])):
            profiles_array.append(sort_resh_profiles_to_plot_T[wave][time].tolist())
        print(profiles_array)
        data_to_save = {
                    'area_with_errors': { 'area' : profiles_array,
                                          'impact_param' : x_impact_param,
                                          'gap' : gaps[wave]}

        }

        data[str(waves[wave])] = data_to_save

    print(data)
    out_dir = r"C:\Users\elena\Documents\GitHub\spec\avantes\out_data"
    path = os.path.join(out_dir, f"area_space_profiles_{waves}_{file_name}.json")

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)



def init_plots_curr_line(data_directory_folder, waves, gaps):

    sort_resh_profiles_to_plot_T, x_impact_param = init_data_by_impact_param(data_directory_folder, waves, gaps)

    save_data_to_json_imp(sort_resh_profiles_to_plot_T, x_impact_param, data_directory_folder, waves, gaps)

    wave_name = 'He I'

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

    x_time = [i * 4 for i in range(len(sort_resh_profiles_to_plot_T[0]))]

    for j in range(len(sort_resh_profiles_to_plot_T)):
        plt.figure()
        plt.title('Линия ' + wave_name + ' ' + str(waves[j]) + ' nm')
        for i in range(len(sort_resh_profiles_to_plot_T[j])):
            plt.plot(x_impact_param, sort_resh_profiles_to_plot_T[j][i], color=colors[i], linestyle=linestyles[i], label=str(x_time[i]+2))

        plt.xlabel('r, mm')
        plt.ylabel('Intesity, a. u.')
        plt.grid(True)

        plt.legend()
    plt.show()


def main():

    data_directory_folder = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\data\260416\\'
    waves = [587.5247, 667.7794, 706.5, 728.0699]
    gaps = [0.4, 0.4, 0.4, 0.4]

    init_plots_curr_line(data_directory_folder, waves, gaps)


main()