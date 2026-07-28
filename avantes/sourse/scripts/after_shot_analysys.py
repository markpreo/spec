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
    data_wo_neg = data_negatives_to_zeros(data)

    times = 20

    mask = np.isin(waves, exact_curr_waves)

    n_waves = len(exact_curr_waves)
    curr_intens = np.zeros((times, n_waves))

    print(curr_intens.shape)
    for i in range(times):
        curr_intens[i] = data_wo_neg[i][mask]

    print(curr_intens)
    return curr_intens, exact_curr_waves, waves, data_wo_neg


def data_negatives_to_zeros(data):
    data = np.asarray(data)
    data[data < 0] = 0
    return data



def gauss(x, A, sigma, A0, d_l, center):
    return A * np.exp(-(x - center - d_l) ** 2 / (2 * sigma ** 2)) + A0


#425.4331, 427.4806, 428.9733, 434.0625, 485.9375

def gauss_sum_of_CIII_OII(x, A1, A2, A3, A4, sigma_CIII, sigma_OII, A0, d_l_CIII, d_l_OII):
    res_gauss_sum_of_CIII_OII = (A1 * np.exp(-(x - 464.728 - d_l_CIII) ** 2 / (2 * sigma_CIII ** 2))) + (
                    A2 * np.exp(-(x - 464.916 - d_l_OII) ** 2 / (2 * sigma_OII ** 2))) + \
                    (A3 * np.exp(-(x - 465.025 - d_l_CIII) ** 2 / (2 * sigma_CIII ** 2))) + (
                    A4 * np.exp(-(x - 465.147 - d_l_CIII) ** 2 / (2 * sigma_CIII ** 2))) + A0
    return res_gauss_sum_of_CIII_OII
# СIII - 464.28, OII - 464.916,* CIII - 465.025, 465.147,


def integrate(A, sigma):
    return A * sigma * np.sqrt(np.pi * 2)


def approx_gauss(A, sigma, A0, d_l, x_region, y_region, wave): #подумать над разными ограничениями, передавать из в функцию?
    model = Model(gauss, independent_vars=['x', 'center'])

    params = model.make_params()
    params.add('area', expr='A*abs(sigma)*sqrt(2*pi)')

    params['A'].set(value=A, min=0, max=2**16)
    params['sigma'].set(value=sigma, min=0, max=2)
    params['A0'].set(value=A0, min=0, max=1000)
    params['d_l'].set(value=d_l, min=-0.5, max=0.5)

    result = model.fit(y_region, params, x=x_region, center=wave)
    print('Площадь с погрешностью')
    print(result.params['area'].value, '+/-', result.params['area'].stderr)

    return result


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
    print(result.params)
    return result





    #одинарные пики отдельно стоящие
def init_popt_data(data_directory, curr_waves, gaps):
    curr_intens, exact_curr_waves, waves, data = data_by_time(data_directory, curr_waves)
    popt_by_time = np.zeros((20, len(exact_curr_waves), 4))
    x_regions_all = []
    y_regions_all = []


    for time in range(20):
        x_regions_all.append([])
        y_regions_all.append([])
        for wave in range(len(exact_curr_waves)):
            peak_region_mask = (waves >= exact_curr_waves[wave] - gaps[wave]) & (waves <= exact_curr_waves[wave] + gaps[wave]) #сделать для каждой линии свое окно
            x_region = waves[peak_region_mask]
            y_region = data[time][peak_region_mask]
            A, sigma, A0, d_l = curr_intens[time][wave], 0.1, 0, 0

            result = approx_gauss(A, sigma, A0, d_l, x_region, y_region, exact_curr_waves[wave])

            p = result.params
            popt_by_time[time][wave] = [p['A'].value, p['sigma'].value,
                                  p['A0'].value, p['d_l'].value]
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

        result = approx_gauss_CIII_OII_sum_lmfit(x_region, y_region,
                                                             A1, A2, A3, A4,
                                                             sigma_CIII, sigma_OII,
                                                             A0,
                                                             d_l_CIII, d_l_OII)
        p = result.params
        popt_by_time[time] = [p['A1'].value, p['A2'].value, p['A3'].value, p['A4'].value,
                                p['sigma_CIII'].value, p['sigma_OII'].value,
                                p['A0'].value,
                                p['d_l_CIII'].value, p['d_l_OII'].value]

        x_regions_all[time].append(x_region)
        y_regions_all[time].append(y_region)

    return popt_by_time, exact_curr_waves, x_regions_all, y_regions_all




def calc_gauss_integral(data_directory, curr_waves, gaps):
    popt_by_time, exact_curr_waves, x_regions_all, y_regions_all = init_popt_data(data_directory, curr_waves, gaps)
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

def visualize_integral_single_gauss(data_directory, curr_waves, gaps):
    res_integrate_by_time = calc_gauss_integral(data_directory, curr_waves, gaps)
    res_integrate_by_time_T = np.transpose(res_integrate_by_time)

    # res_integrate_by_time_CIII_OII = calc_gauss_sum_of_CIII_OII_integral(data_directory, curr_waves)
    # res_integrate_by_time_CIII_OII_T = np.transpose(res_integrate_by_time_CIII_OII)
    colors = ['#3C9E3C', 'blue']

    linestyles = 'solid'

    line_name = ['Cr I', 'Cr I', 'Cr I', 'D gamma', 'D beta']

    #line_name = ['N II']


    for wave in range(len(curr_waves)):
        plt.figure()
        plt.plot(res_integrate_by_time_T[wave] , color=colors[0], linestyle=linestyles, label=str(curr_waves[wave]))
        # plt.plot(res_integrate_by_time_CIII_OII_T[wave], color=colors[wave], linestyle=linestyles[wave],
        #          label=str(curr_waves[wave]))

        plt.xlabel('Time, ms')
        plt.ylabel('Intesity, a. u.')
        plt.grid(True)
        plt.legend()
        out_dir = r"C:\Users\elena\Documents\GitHub\spec\avantes\Graphs\fit_gauss_by_time\single_peak"

        path = os.path.join(out_dir, f"integ_of_single_peak {curr_waves[wave]}.png")

    # out_dir = r"C:\Users\elena\Documents\GitHub\spec\avantes\Graphs\fit_gauss_by_time\sum_CIII_OII"
    #
    # path = os.path.join(out_dir, f"integ_of_each_gauss_CIII_OII_by_time.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()


def visualize_gauss(data_directory, curr_waves):
    #popt_by_time, exact_curr_waves, x_regions_all, y_regions_all = init_popt_data(data_directory, curr_waves)
    popt_by_time_sums, exact_curr_waves_sums, x_regions_all_sums, y_regions_all_sums = init_popt_data_sums_lmfit(data_directory, curr_waves)

    colors = '#3C9E3C'

    linestyles = 'solid'


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
    for time in range(20):
        x_region_fit_sums = np.linspace(min(x_regions_all_sums[time][0]), max(x_regions_all_sums[time][0]), 200)

        plt.plot(x_regions_all_sums[time][0], y_regions_all_sums[time][0], color=colors, linestyle=linestyles,
                 label=str(time))
        print('popt_by_time_sums')
        print(popt_by_time_sums[time])
        plt.plot(x_region_fit_sums,
                gauss(x_region_fit_sums,
                    popt_by_time_sums[time][0],
                    popt_by_time_sums[time][4],
                    popt_by_time_sums[time][6],
                    popt_by_time_sums[time][7],
                    464.728),
                    color='pink', linestyle=linestyles[time])
        plt.plot(x_region_fit_sums,
                gauss(x_region_fit_sums,
                    popt_by_time_sums[time][1],
                    popt_by_time_sums[time][5],
                    popt_by_time_sums[time][6],
                    popt_by_time_sums[time][8],
                    464.916),
                    color='red', linestyle=linestyles[time])
        plt.plot(x_region_fit_sums,
                gauss(x_region_fit_sums,
                    popt_by_time_sums[time][2],
                    popt_by_time_sums[time][4],
                    popt_by_time_sums[time][6],
                    popt_by_time_sums[time][7],
                    465.025),
                    color='violet', linestyle=linestyles[time])
        plt.plot(x_region_fit_sums,
                gauss(x_region_fit_sums,
                    popt_by_time_sums[time][3],
                    popt_by_time_sums[time][4],
                    popt_by_time_sums[time][6],
                    popt_by_time_sums[time][7],
                    465.147),
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

        out_dir = r"C:\Users\elena\Documents\GitHub\spec\avantes\Graphs\fit_gauss_by_time\sum_CIII_OII"

        path = os.path.join(out_dir, f"fit_gauss_t{time:02d}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()


def visualize_single_gauss(data_directory, curr_waves, gaps):
    res_integrate_by_time = calc_gauss_integral(data_directory, curr_waves, gaps)
    res_integrate_by_time_T = np.transpose(res_integrate_by_time)
    popt_by_time, exact_curr_waves, x_regions_all, y_regions_all = init_popt_data(
        data_directory, curr_waves, gaps)

    colors = ['#3C9E3C', 'blue']

    linestyles = 'solid'
    if len(gaps) > 1:
        line_name = ['Cr I', 'Cr I', 'Cr I', 'D gamma', 'D beta']
    else:
        line_name = ['N II']

    for wave in range(len(exact_curr_waves)):
        plt.figure(wave)
        plt.title(line_name[wave] + ' peak')
        for time in range(20):
            print(np.shape(popt_by_time))
            x_region_fit = np.linspace(min(x_regions_all[time][wave]), max(x_regions_all[time][wave]), 200)
            plt.plot(x_regions_all[time][wave], y_regions_all[time][wave], color=colors[0], linestyle=linestyles, label=str(time))
            plt.plot(x_region_fit,
                     gauss(x_region_fit,
                           popt_by_time[time][wave][0],
                           popt_by_time[time][wave][1],
                           popt_by_time[time][wave][2],
                           popt_by_time[time][wave][3],
                           exact_curr_waves[wave]),
                     color=colors[1], linestyle=linestyles, label=f"{line_name[wave]} + {exact_curr_waves[wave]} nm")
            plt.xlabel('Wavelenght, nm')
            plt.ylabel('Intesity, a. u.')
            plt.grid(True)
            plt.legend()

            out_dir = r"C:\Users\elena\Documents\GitHub\spec\avantes\Graphs\fit_gauss_by_time\single_peak"

            path = os.path.join(out_dir, f"integ_of_single_peak_{line_name[wave]}_{curr_waves[wave]}_shot_N{time}.png")

            # out_dir = r"C:\Users\elena\Documents\GitHub\spec\avantes\Graphs\fit_gauss_by_time\sum_CIII_OII"
            #
            # path = os.path.join(out_dir, f"integ_of_each_gauss_CIII_OII_by_time.png")
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()

def main():
    data_directory = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\data\111225\p40 66.STR8'
    data_directory_blue = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\data\111225\p40 45.STR8'

    visualize_single_gauss(data_directory=data_directory, curr_waves=[568.125], gaps=[1.0])
    visualize_single_gauss(data_directory=data_directory_blue, curr_waves=[425.4331, 427.4806, 428.9733, 434.0625, 485.9375], gaps=[0.4, 0.4, 0.4, 0.4, 0.4])
    #curr_waves=[425.4331, 427.4806, 428.9733, 434.0625, 485.9375]

#425.4331, 427.455, 428.9733, 434.062, 434.71, 486.11
main()