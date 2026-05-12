import numpy as np

x1 = -750
y1 = 1981.32
y1 = y1/10

x2 = 750
y2 = 1981.32
y2 = y2/10

Th = 2585
w = 1.823

alpha = Th/w
S = x2 - x1
h = y2 - y1

xv = (x1 + x2) / 2.0 - alpha * np.arcsinh(
    h / (2.0 * alpha * np.sinh(S / (2.0 * alpha))))

#print(xv)

yv = y1 - alpha * (np.cosh((x1 - xv) / alpha) - 1.0)

x = np.linspace(x1,x2,101)
y = yv + alpha * (np.cosh((x - xv) / alpha) - 1.0)
y = y*10

points = np.column_stack((x,y))

with open('ct_pt_gen.txt', 'w') as f:
    for x, y in points:
        f.write(f"{x:.6f},{y:.6f}\n")