import avaread
import matplotlib.pyplot as plt
import os
import numpy as np

def fillBkgs(datas, low_signal_times):
    bkgs = []
    for i in range(low_signal_times):
        bkgs.append(datas.scope.T[i])
    return bkgs

data_directory10 = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\111225\p40 45.STR8'
data_directory11 = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\111225\p40 66.STR8'

datas_blue = avaread.read_file(data_directory10)
datas_red = avaread.read_file(data_directory11)

low_signal_times = 3

bkgs_b = fillBkgs(datas_blue, low_signal_times)
bkgd_b = np.average(bkgs_b, axis=0)

bkgs_r = fillBkgs(datas_red, low_signal_times)
bkgd_r = np.average(bkgs_r, axis=0)

times = 15
waves_b = datas_blue.wavelength
waves_r = datas_red.wavelength
waves = np.append(waves_b, waves_r)

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

for time in range(times):
    final_spectrum_b = datas_blue.scope.T[time] - bkgd_b
    final_spectrum_r = datas_red.scope.T[time] - bkgd_r
    final_spectrum = np.append(final_spectrum_b, final_spectrum_r)

    plt.plot(waves, final_spectrum, color=colors[time], linestyle=linestyles[time], label=str(4*time+2)+ ' ms')

width = 1
colorr = '#FFB6C1'

plt.axvline(x=464.74, color=colorr, linewidth=width)
plt.axvline(x=464.916, color=colorr, linewidth=width)
plt.axvline(x=465.025, color=colorr, linewidth=width)
plt.axvline(x=434.74, color=colorr, linewidth=width)
plt.axvline(x=435.139, color=colorr, linewidth=width)
plt.axvline(x=657.8, color=colorr, linewidth=width)
plt.axvline(x=658.28, color=colorr, linewidth=width)
plt.axvline(x=425.4331, color=colorr, linewidth=width)
plt.axvline(x=427.4806, color=colorr, linewidth=width)
plt.axvline(x=428.9733, color=colorr, linewidth=width)
plt.axvline(x=568.125, color=colorr, linewidth=width)
plt.axvline(x=434.0625, color=colorr, linewidth=width)
plt.axvline(x=485.9375, color=colorr, linewidth=width)
plt.axvline(x=656.25, color=colorr, linewidth=width)


plt.axvspan(551.16, 552.16, alpha=0.3, color='blue')
plt.axvspan(584.15995, 584.55995, alpha=0.3, color='blue')
plt.axvspan(674.8709, 675.5709, alpha=0.3, color='blue')


# plt.axvline(x=464.28, color=colorr, linewidth=width, label='C III 464.28')
# plt.axvline(x=464.916, color=colorr, linewidth=width, label='O II 464.916')
# plt.axvline(x=465.025, color=colorr, linewidth=width, label='C III 465.025')
# plt.axvline(x=434.74, color=colorr, linewidth=width, label='O II 434.74')
# plt.axvline(x=435.139, color=colorr, linewidth=width, label='O II 435.139')
# plt.axvline(x=657.8, color=colorr, linewidth=width, label='C II 657.8')
# plt.axvline(x=658.28, color=colorr, linewidth=width, label='C II 658.28')
# plt.axvline(x=425.4331, color=colorr, linewidth=width, label='Cr I 425.4331')
# plt.axvline(x=427.4806, color=colorr, linewidth=width, label='Cr I 427.4806')
# plt.axvline(x=428.9733, color=colorr, linewidth=width, label='Cr I 428.9733')
# plt.axvline(x=568.125, color=colorr, linewidth=width, label='N II 568.125')
# plt.axvline(x=434.0625, color=colorr, linewidth=width, label='$D\\gamma$ 434.0625')
# plt.axvline(x=485.9375, color=colorr, linewidth=width, label='$D\\beta$ 485.9375')
# plt.axvline(x=656.25, color=colorr, linewidth=width, label='$D\\alpha$ 656.25')

plt.xlabel('wavelenght, nm')
plt.ylabel('Intesity, a. u.')
plt.grid(True)

plt.legend()
plt.show()






