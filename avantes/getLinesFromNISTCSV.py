import pandas as pd
import roman


def getObservedLinesNIST(file):
    lines_dict = {}
    with open(file, 'r') as f:
        f.readline()
        for line in f:
            line = line.split(',')

            if line[2] != '':
                element = line[0] + roman.toRoman(int(line[1]))
                if element not in lines_dict.keys():
                    lines_dict[element] = []
                wl = float(line[2])
                intensity = int(''.join([char for char in line[6] if char.isdigit()])) if any(char.isdigit() for char in line[6]) else 0
                lines_dict[element].append({'wl': wl, 'intensity': intensity, 'stageNumber': int(line[1])})
            elif line[4] != '':
                element = line[0] + roman.toRoman(int(line[1]))
                if element not in lines_dict.keys():
                    lines_dict[element] = []
                wl = float(line[4])
                intensity = 0
                lines_dict[element].append({'wl': wl, 'intensity': intensity, 'stageNumber': int(line[1])})

    return {key: lines_dict[key] for key in sorted(lines_dict)}
