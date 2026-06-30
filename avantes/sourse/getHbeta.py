import avaread
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import resample


def gauss(x, a, b, c):
    return a*np.exp(-(x-b)**2 / (2*c**2))


def gauss2(x, a, b, c, d, e, f):
    return a*np.exp(-(x-b)**2 / (2*c**2)) + d*np.exp(-(x-e)**2 / (2*f**2))


file = r'C:\Users\elena\PycharmProjects\PythonProject\.venv\FTI_work\avantes\111225\p00 20.STR8'

data = avaread.read_file(file)

bkgs = []
bkgs.append(data.scope.T[0])
bkgs.append(data.scope.T[1])
bkgs.append(data.scope.T[2])

bkgd = np.average(bkgs, axis=0)
errors = np.std(bkgs, axis=0)
bkgd_errors = errors / 2**0.5

errors = np.sqrt(np.square(errors) + np.square(bkgd_errors))

spectrum = data.scope.T[5] - bkgd
wavelength = data.wavelength

x = wavelength[1690:1750]
y = spectrum[1690:1750]

popt, pcov = curve_fit(gauss, x, y, p0=[45300, 486, 0.05])
popt2, pcov2 = curve_fit(gauss2, x, y, p0=[45000, 486.1, -0.068, 12000, 486, 0.1])

x_smooth = np.linspace(x[0], x[-1], 1000)

fitted = gauss(x_smooth, *popt)
print(popt)

fitted2 = gauss2(x_smooth, *popt2)
print(popt2)

plt.plot(x, y, label='spectrum')
# plt.plot(x_smooth, fitted, label='fitted1')
plt.plot(x_smooth, fitted2, label='fitted')
plt.plot(x_smooth, gauss(x_smooth, popt2[0], popt2[1], popt2[2]), label='cold')
plt.plot(x_smooth, gauss(x_smooth, popt2[3], popt2[4], popt2[5]), label='hot')
# plt.axvline(486.1, color='r')
# plt.axvline(434.1, color='r')
# plt.axvline(410.2, color='r')
plt.legend()
plt.grid()
plt.show()

integral_cold = np.trapezoid(gauss(x_smooth, popt2[0], popt2[1], popt2[2]), x_smooth)
integral_hot = np.trapezoid(gauss(x_smooth, popt2[3], popt2[4], popt2[5]), x_smooth)

print(integral_cold)
print(integral_hot)
print(integral_cold / integral_hot)

width_cold = 2.355 * np.abs(popt2[2])
width_hot = 2.355 * np.abs(popt2[5])

print(width_cold)
print(width_hot)

Tcold = 2 * (width_cold / (2.43e-3 * popt2[1]))**2 * 1000
Thot = 2 * (width_hot / (2.43e-3 * popt2[4]))**2 * 1000

print(Tcold)
print(Thot)
