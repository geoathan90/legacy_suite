import numpy as np
import matplotlib.pyplot as plt
import scripts.tensions as ts

def ice_load(diam, ice, ice_density=900):
    diam = diam / 1000          # Convert diameter from mm to m
    ice = ice * 25.4 / 1000     # Convert ice thickness from inches to m
    return ice_density * np.pi * (diam*ice + ice**2)

def wind_conversion(wind, gravity=9.810665):
    return np.sqrt(gravity * wind / 0.599072)

def wind_load(diam, wind, ice, gravity, w):
    diam = diam / 1000          # Convert diameter from mm to m
    ice = ice * 25.4 / 1000     # Convert ice thickness from inches to m
    wind = wind_conversion(wind, gravity)  # Convert wind pressure (kg/m^2) to velocity (m/s)
    return 0.599072/gravity*(diam + 2*ice)*wind**2

def total_load(diam, ice, ice_density, wind, gravity, w):
    ice_load_value = ice_load(diam, ice, ice_density)
    wind_load_value = wind_load(diam, wind, ice, gravity, w)
    return np.sqrt((ice_load_value + w)**2 + wind_load_value**2)    

##################

def Vor(H, B_mon, phi, T, alpha, diam, wind, ice, w, ice_density=900, gravity=9.810665):

    wh = wind_load(diam, wind, ice, gravity, w)
    wv = w + ice_load(diam, ice, ice_density)

    return (wh*H*np.cos(alpha/2) - B_mon*np.tan(phi)/2 + 2*T*np.sin(alpha/2))/wv/np.tan(phi)

def deflection_angle(V, H, B_mon, T, alpha, diam, wind, ice,  w, ice_density=900, gravity=9.810665):
    
    wh = wind_load(diam, wind, ice, gravity, w)
    wv = w + ice_load(diam, ice, ice_density)

    return np.degrees(np.arctan((2*T*np.sin(alpha/2) + wh*H)/(wv*V + B_mon/2)))

################

def main():
    
    w = 1.303                # kg/m   !!!!! consider subtracting K constant
    diam = 25.15            # mm    Grosbeak = 25.15, Linnet = 18.31
    ice = 0                 # inches
    ice_density = 900       # kg/m^3
    wind = 20               # kg/m2
    gravity = 9.810665      # m/s^2

    phi = np.radians(35)    # the input in degrees 
    T = 2160                # ruling span tension
    B_mon = 50              # kg     
    #alpha = np.radians(0)   # the input in degrees
    
    H = np.linspace(100,450,351)

    alpha_degrees = [0, 1, 2, 3, 4, 5, 6]

    plt.figure(figsize=(8, 8))

    for alpha_deg in alpha_degrees:
        alpha = np.radians(alpha_deg)

        H_max = 450 - 50*alpha_deg

        H_cut = H[H <= H_max]
        #H_cut = H 

        V = Vor(H_cut, B_mon, phi, T, alpha, diam, wind, ice, w, ice_density, gravity)

        plt.plot(H_cut, V, label=f"alpha = {alpha_deg}°")

        
        print(f"slope for {alpha_deg}°: {wind_load(diam, wind, ice, gravity, w)*np.cos(alpha/2)/(
            ice_load(diam, ice, ice_density)+w)/np.tan(phi):.4f}")

    plt.xlabel("Οριζόντιο")
    plt.ylabel("Κατακόρυφο")
    plt.title("")
    #plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.xlim(100, 500)
    plt.ylim(100, 650)
    plt.yticks(np.arange(100, 650, 50))
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.show()
    plt.savefig("undercuts/plot.png")

    
    #V = Vor(H,B_mon, phi, T, alpha, diam, wind, ice, ice_density, gravity, w)
    
    #plt.plot(H,V)
    
    #print(Vor(145,B_mon, phi, T, alpha, diam, wind, ice, ice_density, gravity, w))
    #rint(Vor(450,B_mon, phi, T, alpha, diam, wind, ice, ice_density, gravity, w))
    
    
    #print(ice_load(diam, ice, ice_density))

    # print(f"slope: {wind_load(diam, wind, ice, gravity, w)*np.cos(alpha/2)/(
    #     ice_load(diam, ice, ice_density)+w)/np.tan(phi)}")
    
    #print(total_load(diam, ice, ice_density, wind, gravity, w))

    #plt.plot([1, 2, 3], [1, 4, 11])
    #plt.savefig("plot.png")


def main2():
    w_original = 1.823                # kg/m   !!!!! consider subtracting K constant
    diam = 30.42            # mm    Grosbeak = 25.15, Linnet = 18.31
    ice = 0                 # inches
    ice_density = 900       # kg/m^3
    wind = 44               # wind pressure kg/m2
    gravity = 9.810665      # m/s^2
    alpha = np.radians(4.5)   # the input in degrees
    B_mon = 0               # kg  -considered zero for safety

    H = 370.0               # measured horizontal load
    Tf_0 = 3151              # ruling span final tension @ 0 degrees
    Tf_40 = 2681             # ruling span final tension @ 40 degrees
    Ti_0 = 3480              # ruling span initial tension @ 0 degrees
    Ti_40 = 2945             # ruling span initial tension @ 40 degrees
    span1 = 525
    span2 = 214.96
    dh1 = 36.02
    dh2 = 19.804
    ruling_span = 420.8288

    Vf_0 = ts.synoliko_katakoryfo(span1, dh1, Tf_0, w_original, span2, dh2, Tf_0, w_original)
    Vf_40 = ts.synoliko_katakoryfo(span1, dh1, Tf_40, w_original, span2, dh2, Tf_40, w_original)    
    Vi_0 = ts.synoliko_katakoryfo(span1, dh1, Ti_0, w_original, span2, dh2, Ti_0, w_original)
    Vi_40 = ts.synoliko_katakoryfo(span1, dh1, Ti_40, w_original, span2, dh2, Ti_40, w_original)

    Ti_neg10 = ts.solve_for_H2(ruling_span, Ti_0, 0, -10, w_original, w_original)
    Ti_50= ts.solve_for_H2(ruling_span, Ti_40, 40, 50, w_original, w_original)    
    

    w_total = total_load(diam, ice, ice_density, wind, gravity, w_original)
    print(f"Total weight = {w_total:.2f} kg/m")



    angle = deflection_angle(V, H, B_mon, T, alpha, diam, wind, ice, w_original, ice_density, gravity)

    print(f"Deflection angle = {angle:.2f}°")

if __name__ == "__main__":
    #main()
    main2()
