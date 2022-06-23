import numpy as np
import matplotlib.pyplot as plt


PVD_IN = np.array((99.474, 99.760, 99.834))
PVD_CLEF = np.array((99.714, 99.88, 99.899))

Apple2020_IN = np.array((48.753, 92.798, 93.939))
Apple2020_CLEF = np.array((95.291, 97.23, 97.521))

Apple2021_IN = np.array((93.174, 93.658, 95.246))
Apple2021_CLEF = np.array((95.506, 95.566, 96.374))

Taiwan_IN = np.array((25.197, 43.307, 55.906))
Taiwan_CLEF = np.array((59.055, 75.591, 81.89))


x = np.array((20, 40, 60))

plt.plot(x, PVD_IN, 'b-*')
plt.plot(x, PVD_CLEF, 'r-o')

plt.plot(x, Apple2020_IN, 'b-<')
plt.plot(x, Apple2020_CLEF, 'r->')

plt.plot(x, Apple2021_IN, 'b-^')
plt.plot(x, Apple2021_CLEF, 'r-_')

plt.plot(x, Taiwan_IN, 'b-*')
plt.plot(x, Taiwan_CLEF, 'r-o')

plt.show()