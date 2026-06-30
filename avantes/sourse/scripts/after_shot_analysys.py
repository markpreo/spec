import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from functools import partial
from scipy.integrate import quad
import avaread
import matplotlib.patches as patches
import math
import os
from lmfit import Model

def init_data(data_directory):
    times = 20
    pre_data = avaread.read_file(data_directory)

    low_signal_times = 3

    bkgs = fill_bkgs(pre_data, low_signal_times)

    waves = pre_data.wavelength

    data = np.array(pre_data.scope)
    data = data.transpose()
    data = data - bkgs

    return data, waves


def fill_bkgs(data, low_signal_times):
    bkgs = []
    for i in range(low_signal_times):
        bkgs.append(data.scope.T[i])
    bkgs = np.array(bkgs)
    bkgs = np.average(bkgs, 0)
    bkgs = bkgs.reshape(1, 2048)
    print(bkgs.shape)
    return bkgs


def waves_search(waves, curr_waves):
    exact_curr_waves = np.array([])
    for c_wave in curr_waves:
        diff = []
        for wave in waves:
            diff.append(abs(wave - c_wave))
        exact_curr_waves = np.append(exact_curr_waves, waves[np.argmin(diff)])

    print(exact_curr_waves)
    return exact_curr_waves


def data_by_time(data_directory, curr_waves):
    print('data_by_time')
    data, waves = init_data(data_directory)
    exact_curr_waves = waves_search(waves, curr_waves)
    times = 20

    mask = np.isin(waves, exact_curr_waves)

    n_waves = len(exact_curr_waves)
    curr_intens = np.zeros((times, n_waves))

    print(curr_intens.shape)
    for i in range(times):
        curr_intens[i] = data[i][mask]

    print(curr_intens)
    return curr_intens, exact_curr_waves, waves, data

def gauss(x, A, sigma, A0, d_l, center):
    return A * np.exp(-(x - center - d_l) ** 2 / (2 * sigma ** 2)) + A0

def gauss_sum_of_CIII_OII(x, A1, A2, A3, A4, sigma_CIII, sigma_OII, A0, d_l_CIII, d_l_OII):
    res_gauss_sum_of_CIII_OII = (A1 * np.exp(-(x - 464.728 - d_l_CIII) ** 2 / (2 * sigma_CIII ** 2))) + (
                    A2 * np.exp(-(x - 464.916 - d_l_OII) ** 2 / (2 * sigma_OII ** 2))) + \
                    (A3 * np.exp(-(x - 465.025 - d_l_CIII) ** 2 / (2 * sigma_CIII ** 2))) + (
                    A4 * np.exp(-(x - 465.147 - d_l_CIII) ** 2 / (2 * sigma_CIII ** 2))) + A0
    return res_gauss_sum_of_CIII_OII
# СIII - 464.28, OII - 464.916,* CIII - 465.025, 465.147,

def integrate(A, sigma):#переписать как 3 буквы вручную гаусс а не численно
    return A * sigma * np.sqrt(np.pi * 2)

# def integrate_CIII_OII_sum(A1, A2, A3, A4, sigma_CIII, sigma_OII, A0, d_l, x_region):
#     def gauss_fixed(x):
#         return gauss_sum_of_CIII_OII(x, A1, A2, A3, A4, sigma_CIII, sigma_OII, A0, d_l)
#
#     l, u = min(x_region), max(x_region)
#     val, err = quad(gauss_fixed, l, u, epsabs=1e-8, epsrel=1e-8, limit=500)
#     return val, err

def approx_gauss(A, sigma, A0, d_l, x_region, y_region, wave):

    curr_A, curr_sigma, curr_A0, curr_d_l = A, sigma, A0, d_l
    fixed_gauss = partial(gauss, center=wave)

    popt, pcov = curve_fit(
        fixed_gauss, x_region, y_region,
        p0=[curr_A, curr_sigma, curr_A0, curr_d_l]#добавить границы и посмотреть ошибки
    )
    return popt


def approx_gauss_CIII_OII_sum(A1, A2, A3, A4, sigma_CIII, sigma_OII, A0, d_l_CIII, d_l_OII, x_region, y_region):

    popt, pcov = curve_fit(
        gauss_sum_of_CIII_OII, x_region, y_region,
        p0=[A1, A2, A3, A4, sigma_CIII, sigma_OII, A0, d_l_CIII, d_l_OII], # и посмотреть ошибки
        bounds=([-50,-50,-50,-50,0,0,0,-0.1, -0.1], [15000, 15000, 15000, 15000, 2, 2, 100, 0.1,0.1]),
        maxfev=20000)

    return popt


def approx_gauss_CIII_OII_sum_lmfit(x_region, y_region,
                                     A1_init, A2_init, A3_init, A4_init,
                                     sigma_CIII_init, sigma_OII_init,
                                     A0_init, d_l_CIII_init, d_l_OII_init):

    model = Model(gauss_sum_of_CIII_OII, independent_vars=['x'])

    params = model.make_params()
    params['A1'].set(value=A1_init, min=-50, max=15000)
    params['A2'].set(value=A2_init, min=-50, max=15000)
    params['A3'].set(value=A3_init, min=-50, max=15000)
    params['A4'].set(value=A4_init, min=-50, max=15000)

    params['sigma_CIII'].set(value=sigma_CIII_init, min=0, max=2)
    params['sigma_OII'].set(value=sigma_OII_init, min=0, max=2)

    params['A0'].set(value=A0_init, min=0, max=100)

    params['d_l_CIII'].set(value=d_l_CIII_init, min=-0.1, max=0.1)
    params['d_l_OII'].set(value=d_l_OII_init, min=-0.1, max=0.1)

    result = model.fit(y_region, params, x=x_region)
    print(result)
    return result





    #одинарные пики отдельно стоящие
def init_popt_data(data_directory, curr_waves):
    curr_intens, exact_curr_waves, waves, data = data_by_time(data_directory, curr_waves)
    popt_by_time = np.zeros((20, len(exact_curr_waves), 4))
    x_regions_all = []
    y_regions_all = []

    for time in range(20):
        x_regions_all.append([])
        y_regions_all.append([])
        for wave in range(len(exact_curr_waves)):
            peak_region_mask = (waves >= exact_curr_waves[wave] - 0.4) & (waves <= exact_curr_waves[wave] + 0.4) #сделать для каждой линии свое окно
            x_region = waves[peak_region_mask]
            y_region = data[time][peak_region_mask]
            A, sigma, A0, d_l = curr_intens[time][wave], 0.1, 0, 0
            popt_by_time[time][wave] = approx_gauss(A, sigma, A0, d_l, x_region, y_region, exact_curr_waves[wave])
            print('x_region')
            print(x_region)
            print('y_region')
            print(y_region)
            x_regions_all[time].append(x_region)
            y_regions_all[time].append(y_region)

    print(popt_by_time)
    # x_regions_all = np.array(x_regions_all)
    # y_regions_all = np.array(y_regions_all)

    return popt_by_time, exact_curr_waves, x_regions_all, y_regions_all

    #суммарные пики хардкод
# def init_popt_data_sums(data_directory, curr_waves):
#     curr_intens, exact_curr_waves, waves, data = data_by_time(data_directory, curr_waves)
#     popt_by_time = np.zeros((20, 9))
#     x_regions_all = []
#     y_regions_all = []
#
#     for time in range(20):
#         x_regions_all.append([])
#         y_regions_all.append([])
#         peak_region_mask = (waves >= exact_curr_waves[0] - 0.4 ) & (waves <= exact_curr_waves[3] + 0.4)  # сделать для каждой линии свое окно
#         x_region = waves[peak_region_mask]
#         y_region = data[time][peak_region_mask]
#
#         A1, A2, A3, A4 = curr_intens[time][0], curr_intens[time][1], curr_intens[time][2], curr_intens[time][3]
#         sigma_CIII, sigma_OII, A0, d_l_CIII, d_l_OII = 0.1, 0.1, 0, 0, 0
#         if time == 15:
#             print('15')
#             print(A1, A2, A3, A4, sigma_CIII, sigma_OII, A0, d_l_CIII, d_l_OII)
#         popt_by_time[time] = approx_gauss_CIII_OII_sum(A1, A2, A3, A4, sigma_CIII, sigma_OII, A0, d_l_CIII, d_l_OII, x_region, y_region)
#         if time == 15:
#             print('popt')
#             print(popt_by_time[15])
#         x_regions_all[time].append(x_region)
#         y_regions_all[time].append(y_region)
#
#     return popt_by_time, exact_curr_waves, x_regions_all, y_regions_all


def init_popt_data_sums_lmfit(data_directory, curr_waves):
    curr_intens, exact_curr_waves, waves, data = data_by_time(data_directory, curr_waves)
    popt_by_time = np.zeros((20, 9))
    x_regions_all = []
    y_regions_all = []

    for time in range(20):
        x_regions_all.append([])
        y_regions_all.append([])
        peak_region_mask = (waves >= exact_curr_waves[0] - 0.4 ) & (waves <= exact_curr_waves[3] + 0.4)  # сделать для каждой линии свое окно
        x_region = waves[peak_region_mask]
        y_region = data[time][peak_region_mask]

        A1, A2, A3, A4 = curr_intens[time][0], curr_intens[time][1], curr_intens[time][2], curr_intens[time][3]
        sigma_CIII, sigma_OII, A0, d_l_CIII, d_l_OII = 0.1, 0.1, 0, 0, 0
        if time == 15:
            print('15')
            print(A1, A2, A3, A4, sigma_CIII, sigma_OII, A0, d_l_CIII, d_l_OII)
        popt_by_time[time] = approx_gauss_CIII_OII_sum_lmfit(x_region, y_region,
                                                             A1, A2, A3, A4,
                                                             sigma_CIII, sigma_OII,
                                                             A0,
                                                             d_l_CIII, d_l_OII)
        if time == 15:
            print('popt')
            print(popt_by_time[15])
        x_regions_all[time].append(x_region)
        y_regions_all[time].append(y_region)

    return popt_by_time, exact_curr_waves, x_regions_all, y_regions_all




def calc_gauss_integral(data_directory, curr_waves):
    popt_by_time, exact_curr_waves, x_regions_all, y_regions_all = init_popt_data(data_directory, curr_waves)
    #print(popt_by_time)
    res_integrate_by_time = np.zeros((20, len(exact_curr_waves)))

    for time in range(20):
        for wave in range(len(exact_curr_waves)):
            #print(popt_by_time[time][wave][0], popt_by_time[time][wave][1], popt_by_time[time][wave][2], exact_curr_waves[wave], x_regions_all[time][wave])
            res_integrate_by_time[time][wave] = integrate(A=popt_by_time[time][wave][0],
                                                          sigma=popt_by_time[time][wave][1])
            print('res_integrate_by_time[time][wave]')
            print(res_integrate_by_time[time][wave])

    print('res_integrate_by_time')
    print(res_integrate_by_time)
    return res_integrate_by_time

def calc_gauss_sum_of_CIII_OII_integral(data_directory, curr_waves):
    popt_by_time, exact_curr_waves, x_regions_all, y_regions_all = init_popt_data_sums_lmfit(data_directory, curr_waves)
    res_integrate_by_time = np.zeros((20, 4))

    for time in range(20):
            res_integrate_by_time[time][0] = integrate(A=popt_by_time[time][0],
                                                          sigma=popt_by_time[time][4])
            res_integrate_by_time[time][1] = integrate(A=popt_by_time[time][1],
                                                       sigma=popt_by_time[time][5])
            res_integrate_by_time[time][2] = integrate(A=popt_by_time[time][2],
                                                       sigma=popt_by_time[time][4])
            res_integrate_by_time[time][3] = integrate(A=popt_by_time[time][3],
                                                       sigma=popt_by_time[time][4])

            print(res_integrate_by_time[time])

    print('res_integrate_by_time')
    print(res_integrate_by_time)
    return res_integrate_by_time

def visualize(data_directory, curr_waves):
    #res_integrate_by_time = calc_gauss_integral(data_directory, curr_waves)
    #res_integrate_by_time_T = np.transpose(res_integrate_by_time)

    res_integrate_by_time_CIII_OII = calc_gauss_sum_of_CIII_OII_integral(data_directory, curr_waves)
    res_integrate_by_time_CIII_OII_T = np.transpose(res_integrate_by_time_CIII_OII)

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
        '#FFCC00',  # 15 - глубокий желтый (золотой)
        '#FFB300',  # 16 - янтарный
        '#FF9900',  # 17 - насыщенный оранжевый
        '#FF7F00',  # 18 - глубокий оранжевый
        '#FF6600',  # 19 - оранжево-красный
        '#FF4500'  # 20 - красно-оранжевый (огненный)
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
        'dotted',  # 15
        'solid',  # 16
        'dashed',  # 17
        'dotted',  # 18
        'solid',  # 19
        'dashed'  # 20
    ]


    for wave in range(len(curr_waves)):
        #plt.plot(res_integrate_by_time_T[wave] , color=colors[wave], linestyle=linestyles[wave], label=str(curr_waves[wave]))
        plt.plot(res_integrate_by_time_CIII_OII_T[wave], color=colors[wave], linestyle=linestyles[wave],
                 label=str(curr_waves[wave]))

    plt.xlabel('Time, ms')
    plt.ylabel('Intesity, a. u.')
    plt.grid(True)
    plt.legend()
    plt.show()
    # потом строим графики

def visualize_gauss(data_directory, curr_waves):
    #popt_by_time, exact_curr_waves, x_regions_all, y_regions_all = init_popt_data(data_directory, curr_waves)
    popt_by_time_sums, exact_curr_waves_sums, x_regions_all_sums, y_regions_all_sums = init_popt_data_sums_lmfit(data_directory, curr_waves)

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
        '#FFCC00',  # 15 - глубокий желтый (золотой)
        '#FFB300',  # 16 - янтарный
        '#FF9900',  # 17 - насыщенный оранжевый
        '#FF7F00',  # 18 - глубокий оранжевый
        '#FF6600',  # 19 - оранжево-красный
        '#FF4500'  # 20 - красно-оранжевый (огненный)
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
        'dotted',  # 15
        'solid',  # 16
        'dashed',  # 17
        'dotted',  # 18
        'solid',  # 19
        'dashed'  # 20
    ]
    # for wave in range(len(exact_curr_waves)):
    #     plt.figure(wave)
    #     plt.title(str(exact_curr_waves[wave]) + ' peak')
    #     for time in range(20):
    #         print(np.shape(popt_by_time))
    #         x_region_fit = np.linspace(min(x_regions_all[time][wave]), max(x_regions_all[time][wave]), 200)
    #         plt.plot(x_regions_all[time][wave], y_regions_all[time][wave], color=colors[time], linestyle=linestyles[time], label=str(time))
    #         plt.plot(x_region_fit,
    #                  gauss(x_region_fit,
    #                        popt_by_time[time][wave][0],
    #                        popt_by_time[time][wave][1],
    #                        popt_by_time[time][wave][2],
    #                        popt_by_time[time][wave][3],
    #                        exact_curr_waves[wave]),
    #                  color='pink', linestyle=linestyles[time], label=str(time))
    #         plt.xlabel('Wavelenght, nm')
    #         plt.ylabel('Intesity, a. u.')
    #         plt.grid(True)
    #         plt.legend()


    plt.figure(4)
    print(exact_curr_waves_sums)
    for time in range(15, 16):
        x_region_fit_sums = np.linspace(min(x_regions_all_sums[time][0]), max(x_regions_all_sums[time][0]), 200)

        plt.plot(x_regions_all_sums[time][0], y_regions_all_sums[time][0], color=colors[time], linestyle=linestyles[time],
                 label=str(time))
        print('popt_by_time_sums')
        print(popt_by_time_sums[time])
        plt.plot(x_region_fit_sums,
                gauss(x_region_fit_sums,
                    popt_by_time_sums[time][0],
                    popt_by_time_sums[time][4],
                    popt_by_time_sums[time][6],
                    popt_by_time_sums[time][7],
                    exact_curr_waves_sums[0]),
                    color='pink', linestyle=linestyles[time])
        plt.plot(x_region_fit_sums,
                gauss(x_region_fit_sums,
                    popt_by_time_sums[time][1],
                    popt_by_time_sums[time][5],
                    popt_by_time_sums[time][6],
                    popt_by_time_sums[time][8],
                    exact_curr_waves_sums[1]),
                    color='red', linestyle=linestyles[time])
        plt.plot(x_region_fit_sums,
                gauss(x_region_fit_sums,
                    popt_by_time_sums[time][2],
                    popt_by_time_sums[time][4],
                    popt_by_time_sums[time][6],
                    popt_by_time_sums[time][7],
                    exact_curr_waves_sums[2]),
                    color='violet', linestyle=linestyles[time])
        plt.plot(x_region_fit_sums,
                gauss(x_region_fit_sums,
                    popt_by_time_sums[time][3],
                    popt_by_time_sums[time][4],
                    popt_by_time_sums[time][6],
                    popt_by_time_sums[time][7],
                    exact_curr_waves_sums[3]),
                    color='purple', linestyle=linestyles[time])

        plt.plot(x_region_fit_sums,
                 gauss_sum_of_CIII_OII(x_region_fit_sums,
                       popt_by_time_sums[time][0],
                       popt_by_time_sums[time][1],
                       popt_by_time_sums[time][2],
                       popt_by_time_sums[time][3],
                       popt_by_time_sums[time][4],
                       popt_by_time_sums[time][5],
                       popt_by_time_sums[time][6],
                       popt_by_time_sums[time][7],
                       popt_by_time_sums[time][8]),
                    color='blue', linestyle=linestyles[time], linewidth=3)


        plt.xlabel('Wavelenght, nm')
        plt.ylabel('Intesity, a. u.')
        plt.grid(True)
        plt.legend()

    plt.show()



def main():
    data_directory = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\data\111225\p40 45.STR8'
    #visualize(data_directory=data_directory, curr_waves=[464.728, 464.916, 465.025, 465.147])
    visualize_gauss(data_directory=data_directory, curr_waves=[464.728, 464.916, 465.025, 465.147])

#425.4331, 427.455, 428.9733, 434.062, 434.71, 486.11
main()