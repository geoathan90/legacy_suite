import tensions as ts
import numpy as np

yo1 = 355.249 # απόλυτο υψόμετρο ανάρτησης κάτω φάσης ΑΔΛ 235|6
yo2 = 350 - 6.465 # απόλυτο υψόμετρο ανάρτησης αγ. προστασίας ΣΑ 744Α

dh1 = -58.80  # υψομετρική διαφορά προς τον 7
dh2 = -37.615 # υψομετρική διαφορά προς τον 743

S1 = 434.57
S2 = 291.38

w1 = 1.823
w2 = 0.46

# αποστάσεις από ΣΧΑ 1005
x1, x2 = 36.8573, 26.5993  # πάνω δεξιά << πιο κρίσιμο
# x1, x2 = 26.2131, 22.0842  # πάνω αριστερά
# x1, x2 = 40.7657, 41.5199  # κάτω δεξιά
# x1, x2 = 30.3191, 37.0582  # κάτω αριστερά

Th1 = [3470, 3330, 3190, 3065, 2945, 2825, 2585]  # 0, 10, 20, 30, 40, 50?, 50/χορδή

Th2 = [1070, 1027.5, 990, 952.5, 917.5, 890, 115000]

y1 = np.zeros(len(Th1))
for i in range(len(Th1)):
    y1[i] = ts.height_at_x(x1,S1,Th1[i],w1,dh1,yo1)
    
y2 = np.zeros(len(Th1))
for i in range(len(Th2)):
    y2[i] = ts.height_at_x(x2,S2,Th2[i],w2,dh2,yo2)

diff = y1 -y2
temps = ["0", "10", "20", "30", "40", "50", "50?/χορδή"]

output = dict(zip(temps, diff))

# 1) Convert NumPy scalars to native Python floats
cleaned = {k: float(v) for k, v in output.items()}

# 2) Print vertically
for k, v in cleaned.items():
    print(f"{k}: {v:.3f}")
