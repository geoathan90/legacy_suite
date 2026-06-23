import numpy as np
import matplotlib.pyplot as plt

Tmax = 1500         # negisti tanysi kg
E = 19.33e9         # metro elastikotitas kg/m2
A = 9.6454e-5       # diatomi m2

spans = np.linspace(100,550,1001)

if __name__ == "__main__":
    print(len(spans))
