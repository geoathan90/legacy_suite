from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline
#from sklearn.isotonic import IsotonicRegression
import sys
import os

###
#                               HOW TO USE:
# (cd into scripts directory), run: "python eval.py [βελοδιάγραμμα] [άνοιγμα1] [άνοιγμα2]"
#  eg. python eval.py 31185 100 450 268.2
# (or any number of values - no commas in between)
###

#csv_path = "data/31185.csv"   # user input later, will probably use sys.argv[1]

diag = str(sys.argv[1])
csv_path = os.path.join("data", diag+".csv")
df = pd.read_csv(csv_path)

####### build interpolators for each temperature

def main():
    interps = {}                       
    for i in range(0,df.shape[1],2):                  # iterate by 2  ||||  range(0,df.shape[1],2)  ----------------------- alternate if more temperatures/conditions
        
        xcol = df.columns[i]                # has to do with the format of the input csv file 
        ycol = df.columns[i+1]              #
        
        suffix = xcol.split("_")[1]         # just get the suffix of the column name (ie drop X in X_20)
        
        xy = df[[xcol,ycol]].dropna().sort_values(xcol).drop_duplicates(subset=xcol, keep="first")      # stock standard data cleaning
        
        x = xy.iloc[:, 0].to_numpy()        
        y = xy.iloc[:, 1].to_numpy()
        
        interps[suffix] = PchipInterpolator(x,y)    # Create the interpolator and store it in the dictionary ||| n X-Y pairs -> n/2 interpolators
        #interps[suffix] = CubicSpline(x,y)         # approach with Cubic Splines; equal performance when interpolating, but much worse when extrapolating
        
        #iso = IsotonicRegression(out_of_bounds='clip')     # Explore a way to more reliably predict out of bounds values
        #interps[suffix] = iso.fit(x, y)                    #



    xs = np.array([float(a) for a in sys.argv[2:]], dtype=float)    # get the input values from command line argument ||| essentially just an 1D array of the inputs

    ys = {k: np.asarray(interp(xs), dtype=float) for k, interp in interps.items()}  # use each interpolator to get the sags for the input spans, store in a dictionary

    if "10" not in interps.keys():                          #
        ys["10"] = (ys["0"]+ys["20"])/2                     #    
    if "30" not in interps.keys():                          #   interpolate/extrapolate the missing temperatures
        ys["30"] = (ys["40"]+ys["20"])/2                    #   
    ys["50"] = ys["40"] + (ys["40"]-ys["20"])/2             #

    if "ICE" in interps.keys():
        rows = ["0", "10", "20", "30", "40", "50", "ICE"]   # rows of the dataframe in order ||| if I have to drop 50, I can just remove it from here and make ICE -> ICE/50
    else: 
        rows=["0", "10", "20", "30", "40", "50"]


    val_list = [ys[k] for k in rows]                        # ys is a dict; val_list is a list of its values, ignoring the keys
    stacked = np.vstack(val_list).T                         # stack the arrays vertically ||| stacked is a 2D array, while val_list is a list of 1D arrays

    data = {}                                               # create a dictionary to hold the data for the dataframe
    for i, x in enumerate(xs):                              
        col_values = stacked[i]                             # col_values is an array of the measured sags
        data[str(x)] = col_values                           # data[str(x)] takes the inputs (the spans) and makes the keys of the dictionary        

    df_out = pd.DataFrame(data, index=rows).round(3)        # main output dataframe

    print(df_out.to_string())

###############  Helpers  ####################

#def stili(col_number, *, _df=df_out):                                     #  use stili(1) to print first column only, stili(2) for second column, etc.
#    return print(_df.iloc[:, col_number-1].to_string(index=False))

if __name__ == "__main__":
    main()  

#######   if I want to SAVE to a csv file
#
#  df_out.to_csv("output.csv")
#
#######

###
#                               HOW TO USE:
# (cd into scripts directory), run: "python eval.py [βελοδιάγραμμα] [άνοιγμα1] [άνοιγμα2]"
#  eg. python eval.py 31185 100 450 268.2
# (or any number of values - no commas in between)
###

