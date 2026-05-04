import tensions as ts
import numpy as np
from eval import evaluate

yo1 = 51.99 + 7.80 # απόλυτο υψόμετρο ανάρτησης άνω φάσης ΤΑΠ 88Ν|1
yo2 = 51.99 + 3.90 # απόλυτο υψόμετρο ανάρτησης μεσαίας φάσης ΤΑΠ 88Ν|1

dh1 = -0.12  # υψομετρική διαφορά προς ΤΑΠ 88Ν (άνω φάση)
dh2 = -0.12 # υψομετρική διαφορά προς ΤΑΠ 88Ν (μεσαία φάση)

S1 = 23.63 #24.56
S2 = 21.3 #22.36

w1 = 1.303
w2 = 1.303

# αποστάσεις ξεκινώντας από ΤΑΠ 88Ν|1
x1, x2 = 11.46, 11.38 #8.17, 8.14 

#sags1 = np.zeros(6)

sags = evaluate("31163",[S1, S2])

sags1 = sags.iloc[:,0]
sags2 = sags.iloc[:,1]

# sags1 = [0.552, 0.592, 0.629, 0.664, 0.702, 0.738, 0.738] # 0, 10, 20, 30, 40, 50?
# sags2 = [0.525, 0.564, 0.600, 0.638, 0.675, 0.713, 0.001]

Th1 = np.zeros(len(sags1))
for i in range(len(sags1)):
    Th1[i] = ts.Th_from_sag(sags1[i],S1,w1)

Th2 = np.zeros(len(sags2))
for i in range(len(sags2)):
    Th2[i] = ts.Th_from_sag(sags2[i],S2,w2)

Th1 = Th2 

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
