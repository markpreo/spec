from raw8 import Raw8
from matplotlib import pyplot as plt

path1 = r"D:\home\projects\SPECTR\avantes\data\241204\CRU00220022.RAW8"
path = r"D:\home\projects\SPECTR\avantes\data\241204\CRU00250025.RAW8"
file = Raw8(path)
file1 = Raw8(path1)

plt.plot(file.getWavelength(), file.getScope()-file1.getScope())
plt.grid()
plt.show()
