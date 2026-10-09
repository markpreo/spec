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

def init_data(data_directory):
    global times
    times = 15
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
    bkgs = bkgs.reshape(1, len(data.scope.T[1]))
    print(bkgs.shape)
    return bkgs


def waves_search(waves, curr_waves):
    exact_curr_waves = np.array([])
    for c_wave in curr_waves:
        diff = []
        for wave in waves:
            diff.append(abs(wave - c_wave))

        exact_curr_waves = np.append(exact_curr_waves, waves[np.argmin(diff, axis=0)])
    print('exact')
    print(exact_curr_waves)
    return exact_curr_waves


def data_by_time(data_directory, curr_waves):
    print('data_by_time')
    data, waves = init_data(data_directory)
    exact_curr_waves = waves_search(waves, curr_waves)
    data_wo_neg = data_negatives_to_zeros(data)



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


def gauss_sum_of_CII(x, A1, A2, sigma_CII, A0, d_l_CII):
    res_gauss_sum_of_CII = (A1 * np.exp(-(x - 657.8 - d_l_CII) ** 2 / (2 * sigma_CII ** 2))) + (
                    A2 * np.exp(-(x - 658.28 - d_l_CII) ** 2 / (2 * sigma_CII ** 2))) + A0
    return res_gauss_sum_of_CII


def gauss_sum_of_OII(x, A1, A2, sigma_OII, A0, d_l_OII):
    res_gauss_sum_of_OII = (A1 * np.exp(-(x - 434.740 - d_l_OII) ** 2 / (2 * sigma_OII ** 2))) + (
                    A2 * np.exp(-(x - 435.139 - d_l_OII) ** 2 / (2 * sigma_OII ** 2))) + A0
    return res_gauss_sum_of_OII


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

    return result


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

    return result


def approx_gauss_CII_group_lmfit(x_region, y_region,
                                     A1_init, A2_init,
                                     sigma_CII_init,
                                     A0_init, d_l_CII_init):

    model = Model(gauss_sum_of_CII, independent_vars=['x'])

    params = model.make_params()

    params.add('area_1', expr='A1*abs(sigma_CII)*sqrt(2*pi)')
    params.add('area_2', expr='A2*abs(sigma_CII)*sqrt(2*pi)')

    params['A1'].set(value=A1_init, min=-50, max=15000)
    params['A2'].set(value=A2_init, min=-50, max=15000)

    params['sigma_CII'].set(value=sigma_CII_init, min=0, max=2)

    params['A0'].set(value=A0_init, min=0, max=100)

    params['d_l_CII'].set(value=d_l_CII_init, min=-0.1, max=0.1)

    result = model.fit(y_region, params, x=x_region)

    return result


def approx_gauss_OII_group_lmfit(x_region, y_region,
                                 A1_init, A2_init,
                                 sigma_OII_init,
                                 A0_init, d_l_OII_init):
    model = Model(gauss_sum_of_OII, independent_vars=['x'])

    params = model.make_params()

    params.add('area_1', expr='A1*abs(sigma_OII)*sqrt(2*pi)')
    params.add('area_2', expr='A2*abs(sigma_OII)*sqrt(2*pi)')

    params['A1'].set(value=A1_init, min=-50, max=15000)
    params['A2'].set(value=A2_init, min=-50, max=15000)

    params['sigma_OII'].set(value=sigma_OII_init, min=0, max=2)

    params['A0'].set(value=A0_init, min=0, max=100)

    params['d_l_OII'].set(value=d_l_OII_init, min=-0.1, max=0.1)

    result = model.fit(y_region, params, x=x_region)

    return result


def nofitting_read_file(file_name):
    input_dir = r"C:\Users\elena\Documents\GitHub\spec\avantes\out_data"

    path = os.path.join(input_dir, file_name)
    print(path)

    with open(path, "r") as file:
        data = json.load(file)

    return data


   # вычленить по нужным х нужные у
    # посчитать ошибку площади для каждой линии по формуле
    # сохранить в нужном порядке



def save_fitting_data_to_json(data3d, area3d, exact_curr_waves, data_directory):
    print('save_nofitting_data_to_json!!!!!!!!!!!!!!!!!!')
    file_name = data_directory[-11:-5]
    data = {}
    for wave in range(len(exact_curr_waves)):
        A_array = []
        sigma_array = []
        A0_array = []
        d_l_array = []
        area_array = []
        stderr_array = []

        for time in range(times):
            A_array.append(data3d[time][wave][0])
            sigma_array.append(data3d[time][wave][1])
            A0_array.append(data3d[time][wave][2])
            d_l_array.append(data3d[time][wave][3])
            area_array.append(area3d[time][wave][0])
            stderr_array.append(area3d[time][wave][1])

        data_to_save = {
                    'params': {
                                'A': A_array,
                                'sigma': sigma_array,
                                'A0': A0_array,
                                'd_l': d_l_array
                    },
                    'area_with_errors': { 'area' : area_array,
                                          'stderr' : stderr_array}

        }

        data[str(exact_curr_waves[wave])] = data_to_save

    out_dir = r"C:\Users\elena\Documents\GitHub\spec\avantes\out_data"
    path = os.path.join(out_dir, f"params_full_frames_{exact_curr_waves}_{file_name}.json")

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)



def save_nofitting_data_to_json(area3d, exact_curr_waves, data_directory):
    print('save_nofitting_data_to_json !!!!!!!!!!!!')
    file_name = data_directory[-11:-5]
    print(file_name)
    data = {}
    for wave in range(len(exact_curr_waves)):
        area_array = []
        stderr_array = []
        for time in range(times):
            area_array.append(area3d[time][wave][0])
            stderr_array.append(area3d[time][wave][1])

        data_to_save = {
                    'area_with_errors': { 'area' : area_array,
                                          'stderr' : stderr_array}

        }

        data[str(exact_curr_waves[wave])] = data_to_save

    out_dir = r"C:\Users\elena\Documents\GitHub\spec\avantes\out_data"
    path = os.path.join(out_dir, f"area_full_frames_{exact_curr_waves}_{file_name}.json")
    print(data)

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)



    #одинарные пики отдельно стоящие

def init_popt_data(data_directory, curr_waves, gaps):
    curr_intens, exact_curr_waves, waves, data = data_by_time(data_directory, curr_waves)
    popt_by_time = np.zeros((times, len(exact_curr_waves), 4))
    integ_by_time = np.zeros((times, len(exact_curr_waves), 2))
    x_regions_all = []
    y_regions_all = []


    for time in range(times):
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
            integ_by_time[time][wave] = [p['area'].value, p['area'].stderr]

            x_regions_all[time].append(x_region)
            y_regions_all[time].append(y_region)

    # x_regions_all = np.array(x_regions_all)
    # y_regions_all = np.array(y_regions_all)
    save_fitting_data_to_json(popt_by_time, integ_by_time, exact_curr_waves, data_directory)

    return popt_by_time, integ_by_time, exact_curr_waves, x_regions_all, y_regions_all


def init_popt_data_sums_lmfit(data_directory, curr_waves):
    curr_intens, exact_curr_waves, waves, data = data_by_time(data_directory, curr_waves)
    popt_by_time = np.zeros((20, 9))
    x_regions_all = []
    y_regions_all = []

    for time in range(20):
        x_regions_all.append([])
        y_regions_all.append([])
        peak_region_mask = (waves >= exact_curr_waves[0] - 0.4 ) & (waves <= exact_curr_waves[3] + 0.4)
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


def init_popt_data_CII_group(data_directory, curr_waves, gaps):
    curr_intens, exact_curr_waves, waves, data = data_by_time(data_directory, curr_waves)
    popt_by_time = np.zeros((20, 2, 4))
    integ_by_time = np.zeros((20, 2, 2))
    x_regions_all = []
    y_regions_all = []


    for time in range(20):
            x_regions_all.append([])
            y_regions_all.append([])
            peak_region_mask = (waves >= exact_curr_waves[0] - gaps[0]) & (waves <= exact_curr_waves[1] + gaps[1])
            x_region = waves[peak_region_mask]
            y_region = data[time][peak_region_mask]
            A1, A2, sigma_CII, A0, d_l_CII = curr_intens[time][0], curr_intens[time][1], 0.1, 0, 0

            result = approx_gauss_CII_group_lmfit(x_region, y_region, A1, A2, sigma_CII, A0, d_l_CII)

            p = result.params
            popt_by_time[time][0] = [p['A1'].value, p['sigma_CII'].value,
                                  p['A0'].value, p['d_l_CII'].value]
            popt_by_time[time][1] = [p['A2'].value, p['sigma_CII'].value,
                                  p['A0'].value, p['d_l_CII'].value]
            integ_by_time[time][0] = [p['area_1'].value, p['area_1'].stderr]
            integ_by_time[time][1] = [p['area_2'].value, p['area_2'].stderr]

            x_regions_all[time].append(x_region)
            y_regions_all[time].append(y_region)


    save_data_to_json(popt_by_time, integ_by_time, exact_curr_waves)

    return popt_by_time, integ_by_time, exact_curr_waves, x_regions_all, y_regions_all



def init_popt_data_OII_group(data_directory, curr_waves, gaps):
    curr_intens, exact_curr_waves, waves, data = data_by_time(data_directory, curr_waves)
    popt_by_time = np.zeros((20, 2, 4))
    integ_by_time = np.zeros((20, 2, 2))
    x_regions_all = []
    y_regions_all = []


    for time in range(20):
            x_regions_all.append([])
            y_regions_all.append([])
            peak_region_mask = (waves >= exact_curr_waves[0] - gaps[0]) & (waves <= exact_curr_waves[1] + gaps[1])
            x_region = waves[peak_region_mask]
            y_region = data[time][peak_region_mask]
            A1, A2, sigma_OII, A0, d_l_OII = curr_intens[time][0], curr_intens[time][1], 0.1, 0, 0

            result = approx_gauss_OII_group_lmfit(x_region, y_region, A1, A2, sigma_OII, A0, d_l_OII)

            p = result.params
            popt_by_time[time][0] = [p['A1'].value, p['sigma_OII'].value,
                                  p['A0'].value, p['d_l_OII'].value]
            popt_by_time[time][1] = [p['A2'].value, p['sigma_OII'].value,
                                  p['A0'].value, p['d_l_OII'].value]
            integ_by_time[time][0] = [p['area_1'].value, p['area_1'].stderr]
            integ_by_time[time][1] = [p['area_2'].value, p['area_2'].stderr]

            x_regions_all[time].append(x_region)
            y_regions_all[time].append(y_region)


    save_data_to_json(popt_by_time, integ_by_time, exact_curr_waves)

    return popt_by_time, integ_by_time, exact_curr_waves, x_regions_all, y_regions_all


# def calc_gauss_integral(data_directory, curr_waves, gaps):
#     popt_by_time, integ_by_time, exact_curr_waves, x_regions_all, y_regions_all = init_popt_data(data_directory, curr_waves, gaps)
#     #print(popt_by_time)
#     res_integrate_by_time = np.zeros((20, len(exact_curr_waves)))
#
#     for time in range(20):
#         for wave in range(len(exact_curr_waves)):
#             #print(popt_by_time[time][wave][0], popt_by_time[time][wave][1], popt_by_time[time][wave][2], exact_curr_waves[wave], x_regions_all[time][wave])
#             res_integrate_by_time[time][wave] = integrate(A=popt_by_time[time][wave][0],
#                                                           sigma=popt_by_time[time][wave][1])
#             print('res_integrate_by_time[time][wave]')
#             print(res_integrate_by_time[time][wave])
#
#     print('res_integrate_by_time')
#     print(res_integrate_by_time)
#     return res_integrate_by_time



def calc_peaks_area_wo_integ(data_directory, curr_waves, gaps, file_name):
    curr_intens, exact_curr_waves, waves, data = data_by_time(data_directory, curr_waves)
    std = nofitting_read_file(file_name)
    std = np.array(std)

    peaks_area_wo_integ = np.zeros([times, len(exact_curr_waves), 2])
    x_regions_all = []
    y_regions_all = []
    print(np.shape(std))

    for time in range(times):
        x_regions_all.append([])
        y_regions_all.append([])
        for wave in range(len(exact_curr_waves)):
            peak_region_mask = (waves >= exact_curr_waves[wave] - gaps[wave]) & (waves <= exact_curr_waves[wave] + gaps[wave]) #сделать для каждой линии свое окно
            x_region = waves[peak_region_mask]
            y_region = data[time][peak_region_mask]
            std_region = std[peak_region_mask]
            peak_area = 0
            peak_error = 0
            for x in range(1, len(x_region)-1):
                peak_area += y_region[x] * 0.5 * (x_region[x+1] - x_region[x-1])
                peak_error += (std_region[x] * 0.5 * (x_region[x + 1] - x_region[x - 1])) ** 2


            peaks_area_wo_integ[time][wave] = peak_area, np.sqrt(peak_error)

            x_regions_all[time].append(x_region)
            y_regions_all[time].append(y_region)

    return peaks_area_wo_integ


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

def vis_peaks_area_wo_integ(data_directory, curr_waves, gaps, file_name):
    peaks_area_wo_integ = calc_peaks_area_wo_integ(data_directory, curr_waves, gaps, file_name)

    save_nofitting_data_to_json(peaks_area_wo_integ, curr_waves, data_directory)

    times_l = [i for i in range(times)]
    peaks_area_wo_integ_T = np.swapaxes(peaks_area_wo_integ, 0, 2)

    colors = ['#3C9E3C', 'blue']

    linestyles = 'solid'

    if len(gaps) > 1:
        line_name = ['He I', 'He I', 'He I', 'He I']
    else:
        line_name = ['N II']

    print(np.shape(peaks_area_wo_integ_T))

    for wave in range(len(curr_waves)):
        plt.figure()
        plt.title(line_name[wave] + ' peak')
        y = peaks_area_wo_integ_T[0][wave]
        err = peaks_area_wo_integ_T[1][wave]
        print('for err ')
        print(times_l, err)

        plt.plot(times_l,
                 y,
                 color=colors[1],
                 linestyle=linestyles,
                 label=f"{line_name[wave]} + {curr_waves[wave]} nm")

        max_y = np.max(y)
        reasonable_max_err = max_y * 0.5
        err = np.where(err > reasonable_max_err, 0, err)

        plt.fill_between(times_l,
                         y + err,
                         y - err,
                         color='blue', alpha=0.3)
        for time in times_l:
            if (err[time] <= 0) or (math.isnan(err[time])):
                plt.scatter(time, y[time], color='red')
            else:
                plt.scatter(time, y[time], color='blue')



        plt.xlabel('Frame, n')
        plt.ylabel('Area, a. u.')
        plt.grid(True)
        plt.legend()

        # out_dir = r"C:\Users\elena\Documents\GitHub\spec\avantes\Graphs\fit_gauss_by_time\single_peak"
        # path = os.path.join(out_dir, f"integ_of_single_peak_{line_name[wave]}_{exact_curr_waves[wave]}.png")
        # plt.savefig(path, dpi=300, bbox_inches="tight")

        plt.show()

def vis_integ_sum_of_CIII_OII_lmfit(data_directory, curr_waves):
    res_integrate_by_time = calc_gauss_sum_of_CIII_OII_integral(data_directory, curr_waves)

    times = [i for i in range(20)]
    sum_of_all = []
    for time in range(20):
        sum_of_all.append(sum(res_integrate_by_time[time][:4]))

    plt.plot(times, sum_of_all, color='b')
    plt.show()

def vis_integ_sum_of_CII_group_lmfit(data_directory, curr_waves, gaps):
    popt_by_time, integ_by_time, exact_curr_waves, x_regions_all, y_regions_all = init_popt_data_CII_group(data_directory, curr_waves, gaps)
    print(exact_curr_waves)
    times = [i for i in range(20)]
    integ_by_time_T = np.swapaxes(integ_by_time, 0, 2)

    line_name = ['C II', 'C II']

    for wave in range(len(exact_curr_waves)):
        plt.figure()
        plt.title(line_name[wave] + ' peak')
        y = integ_by_time_T[0][wave]
        err = integ_by_time_T[1][wave]
        print('for err ')
        print(times, err)

        plt.plot(times,
                 y,
                 color='b',
                 linestyle='solid',
                 label=f"{line_name[wave]} + {exact_curr_waves[wave]} nm")

        max_y = np.max(y)
        reasonable_max_err = max_y * 0.5
        err = np.where(err > reasonable_max_err, 0, err)

        plt.fill_between(times,
                         y + err,
                         y - err,
                         color='blue', alpha=0.3)
        for time in times:
            if (err[time] <= 0) or (math.isnan(err[time])):
                plt.scatter(time, y[time], color='red')
            else:
                plt.scatter(time, y[time], color='blue')



        plt.xlabel('Frame, n')
        plt.ylabel('Area, a. u.')
        plt.grid(True)
        plt.legend()
        plt.show()


def vis_integ_sum_of_OII_group_lmfit(data_directory, curr_waves, gaps):
    popt_by_time, integ_by_time, exact_curr_waves, x_regions_all, y_regions_all = init_popt_data_OII_group(data_directory, curr_waves, gaps)
    print(exact_curr_waves)
    times = [i for i in range(20)]
    integ_by_time_T = np.swapaxes(integ_by_time, 0, 2)

    line_name = ['O II', 'O II']

    for wave in range(len(exact_curr_waves)):
        plt.figure()
        plt.title(line_name[wave] + ' peak')
        y = integ_by_time_T[0][wave]
        err = integ_by_time_T[1][wave]
        print('for err ')
        print(times, err)

        plt.plot(times,
                 y,
                 color='b',
                 linestyle='solid',
                 label=f"{line_name[wave]} + {exact_curr_waves[wave]} nm")

        max_y = np.max(y)
        reasonable_max_err = max_y * 0.5
        err = np.where(err > reasonable_max_err, 0, err)

        plt.fill_between(times,
                         y + err,
                         y - err,
                         color='blue', alpha=0.3)
        for time in times:
            if (err[time] <= 0) or (math.isnan(err[time])):
                plt.scatter(time, y[time], color='red')
            else:
                plt.scatter(time, y[time], color='blue')



        plt.xlabel('Frame, n')
        plt.ylabel('Area, a. u.')
        plt.grid(True)
        plt.legend()
        plt.show()


def visualize_integral_single_gauss(data_directory, curr_waves, gaps):
    res_integrate_by_time = calc_gauss_integral(data_directory, curr_waves, gaps)
    res_integrate_by_time_T = np.transpose(res_integrate_by_time)

    # res_integrate_by_time_CIII_OII = calc_gauss_sum_of_CIII_OII_integral(data_directory, curr_waves)
    # res_integrate_by_time_CIII_OII_T = np.transpose(res_integrate_by_time_CIII_OII)
    colors = ['#3C9E3C', 'blue']

    linestyles = 'solid'

    if len(gaps) > 1:
        line_name = ['Cr I', 'Cr I', 'Cr I', 'D gamma', 'D beta']
    else:
        line_name = ['N II']


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


def vis_integ_lmfit(integ_by_time, gaps, exact_curr_waves):
    times_l = [i for i in range(times)]
    integ_by_time_T = np.swapaxes(integ_by_time, 0, 2)

    colors = ['#3C9E3C', 'blue']

    linestyles = 'solid'

    if len(gaps) > 1:
        line_name = ['He I', 'He I', 'He I', 'He I']
        # line_name = ['Cr I', 'Cr I', 'Cr I', 'D gamma', 'D beta']
    else:
        line_name = ['N II']

    print(np.shape(integ_by_time_T))

    for wave in range(len(exact_curr_waves)):
        plt.figure()
        plt.title(line_name[wave] + ' peak')
        y = integ_by_time_T[0][wave]
        err = integ_by_time_T[1][wave]
        print('for err ')
        print(times, err)

        plt.plot(times_l,
                 y,
                 color=colors[1],
                 linestyle=linestyles,
                 label=f"{line_name[wave]} + {exact_curr_waves[wave]} nm")

        max_y = np.max(y)
        reasonable_max_err = max_y * 0.5
        err = np.where(err > reasonable_max_err, 0, err)

        plt.fill_between(times,
                         y + err,
                         y - err,
                         color='blue', alpha=0.3)
        for time in times_l:
            if (err[time] <= 0) or (math.isnan(err[time])):
                plt.scatter(time, y[time], color='red')
            else:
                plt.scatter(time, y[time], color='blue')



        plt.xlabel('Frame, n')
        plt.ylabel('Area, a. u.')
        plt.grid(True)
        plt.legend()

        # out_dir = r"C:\Users\elena\Documents\GitHub\spec\avantes\Graphs\fit_gauss_by_time\single_peak"
        # path = os.path.join(out_dir, f"integ_of_single_peak_{line_name[wave]}_{exact_curr_waves[wave]}.png")
        # plt.savefig(path, dpi=300, bbox_inches="tight")

        plt.show()


def vis_single_gauss_lmfit(popt_by_time, gaps, exact_curr_waves, x_regions_all, y_regions_all):

    colors = ['#3C9E3C', 'blue']

    linestyles = 'solid'
    if len(gaps) > 1:
        line_name = ['Cr I', 'Cr I', 'Cr I', 'D gamma', 'D beta']
    else:
        line_name = ['N II']

    for wave in range(len(exact_curr_waves)):
        for time in range(20):
            plt.figure()
            plt.title(line_name[wave] + ' peak')
            print(np.shape(popt_by_time))
            x_region_fit = np.linspace(min(x_regions_all[time][wave]), max(x_regions_all[time][wave]), 200)
            plt.plot(x_regions_all[time][wave], y_regions_all[time][wave], color=colors[0], linestyle=linestyles,
                     label=str(time))
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

            #out_dir = r"C:\Users\elena\Documents\GitHub\spec\avantes\Graphs\fit_gauss_by_time\single_peak"

            #path = os.path.join(out_dir, f"gauss_of_single_peak_{line_name[wave]}_{exact_curr_waves[wave]}_shot_N{time}.png")

            # out_dir = r"C:\Users\elena\Documents\GitHub\spec\avantes\Graphs\fit_gauss_by_time\sum_CIII_OII"
            #
            # path = os.path.join(out_dir, f"integ_of_each_gauss_CIII_OII_by_time.png")
            #plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.show()
            #plt.close()


def visualize(data_directory, curr_waves, gaps):

    group_waves = [
        464.728,
        464.916,
        465.025,
        465.147,
    ]

    group_CII = [
        657.8,
        658.28
    ]

    group_OII = [
        434.740,
        435.139
    ]

    #сомнительная логика выбора обрпботки
    if 464.728 in curr_waves:
        vis_integ_sum_of_CIII_OII_lmfit(
            data_directory,
            group_waves
        )

        curr_waves = [
            wave
            for wave in curr_waves
            if wave not in group_waves
        ]


    if 657.8 in curr_waves:
        vis_integ_sum_of_CII_group_lmfit(data_directory, curr_waves, gaps)

        curr_waves = [
            wave
            for wave in curr_waves
            if wave not in group_CII
        ]

    if 434.740 in curr_waves:
        vis_integ_sum_of_OII_group_lmfit(ata_directory, curr_waves, gaps)

        curr_waves = [
            wave
            for wave in curr_waves
            if wave not in group_OII
        ]

    if curr_waves != []:
        popt_by_time, integ_by_time, exact_curr_waves, x_regions_all, y_regions_all = init_popt_data(
            data_directory, curr_waves, gaps)

        vis_integ_lmfit(integ_by_time, gaps, exact_curr_waves)
        #vis_single_gauss_lmfit(popt_by_time, gaps, exact_curr_waves, x_regions_all, y_regions_all)


def main():
    data_directory = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\data\260416\p00 34.STR8'
    data_directory_blue = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\data\111225\p40 45.STR8'


    # visualize(data_directory=data_directory,
    #         curr_waves=[587.5247, 667.7794, 706.5, 728.0699],
    #         gaps=[0.4, 0.4, 0.4, 0.4])
    #curr_waves=[425.4331, 427.4806, 428.9733, 434.0625, 485.9375]

    vis_peaks_area_wo_integ(data_directory=data_directory,
                            curr_waves=[587.5247, 667.7794, 706.5, 728.0699],
                            gaps=[0.4, 0.4, 0.4, 0.4],
                            file_name='errors_red_Noise red absolute.json')

#425.4331, 427.455, 428.9733, 434.062, 434.71, 486.11
#main()