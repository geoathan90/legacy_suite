import scripts.tensions as ts
import pandas as pd
import numpy as np
from scripts.eval import evaluate
import sys
import os
import openpyxl
from pathlib import Path

#usage: print(evaluate(52740,[200,450]))

default_weight = 1.823
w = default_weight

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "larisa1_v2.csv")

diags={"term":"31185","BA350":"31187","BA500":"31188","2000":"52740","1000":"31189"}

loads = {"S5":600, "S5+8.00":600, "G5":700, "G5+8.00":700, "R5":800, "R5+8.00":800, 
         "R5+18.00":800, "RE5":1000, "RE5+8.00":1000, "T5":1000, "T5+8.00":1000, "T5+18.00":1000,
         "TE5":1200, "TE5+8.00":1200, "TE5+18.00":1200, "Z5":800, "Z5+8.00":800, "ZE5":960, "ZE5+8.00":960, "Z5*":800,"Z5+8.00*":800}

df["max_load_kg"] = 0.0
df['sag_0'] = 50.0
df['tensions_0'] = 1000

for i in range(len(df)):
    
    sign2 = df.iloc[i]["type"]
    df.loc[i,"max_load_kg"] =loads[sign2]*2*2.2662 # 2.2662 είναι το βάρος για 0" πάγο και 9# αέρα

    sign = df.iloc[i]["span_type"]
    df2=evaluate(diags[sign],[df.iloc[i]['span']])
    df.loc[i,'sag_0']=float(df2.iloc[0][str(df.iloc[i]['span'])])

df['tensions_0'] = [
    ts.Th_from_sag(sag, span, w)
    for sag, span in zip(df['sag_0'], df['span'])
]

#example df.loc[df["condition_col"] == "match_value", "target_col"] = "new_value"

df.loc[df["span_type"] == "BA350", "tensions_0"] = 3470   #2585 @50C
df.loc[df["span_type"] == "BA500", "tensions_0"] = 3105   #2585 @50C

df['katakoryfo_0'] = ts.synoliko_katakoryfo(df["span"].shift(1).to_numpy(), df["height_diff"].shift(1).to_numpy(), df["tensions_0"].shift(1).to_numpy(), w,
                                          df["span"].to_numpy(), df["height_diff"].to_numpy(), df["tensions_0"].to_numpy(), w)

df['load_percentage_0'] = df['katakoryfo_0']*2*w/df['max_load_kg']*100 
df['load_percentage_0'] = df['load_percentage_0'].round(2)


solve_for_H2_vectorized = np.vectorize(ts.solve_for_H2)

df['tensions_-10'] = solve_for_H2_vectorized(df['span'].to_numpy(), df['tensions_0'].to_numpy(), 0, -10, w, w)

df['katakoryfo_-10'] = ts.synoliko_katakoryfo(df["span"].shift(1).to_numpy(), df["height_diff"].shift(1).to_numpy(), df["tensions_-10"].shift(1).to_numpy(), w,
                                          df["span"].to_numpy(), df["height_diff"].to_numpy(), df["tensions_-10"].to_numpy(), w)

df['load_percentage_-10'] = df['katakoryfo_-10']*2*w/df['max_load_kg']*100 
df['load_percentage_-10'] = df['load_percentage_-10'].round(2)


df['tensions_-10_ICE'] = solve_for_H2_vectorized(df['span'].to_numpy(), df['tensions_0'].to_numpy(), 0, -10, w, 2.5)

df['katakoryfo_-10_ICE'] = ts.synoliko_katakoryfo(df["span"].shift(1).to_numpy(), df["height_diff"].shift(1).to_numpy(), df["tensions_-10_ICE"].shift(1).to_numpy(), 2.5,
                                          df["span"].to_numpy(), df["height_diff"].to_numpy(), df["tensions_-10_ICE"].to_numpy(), 2.5)

df['load_percentage_-10_ICE'] = df['katakoryfo_-10_ICE']*2*2.5/df['max_load_kg']*100 
df['load_percentage_-10_ICE'] = df['load_percentage_-10_ICE'].round(2)



df['tensions_0_ICE'] = solve_for_H2_vectorized(df['span'].to_numpy(), df['tensions_0'].to_numpy(), 0, 0, w, 2.5)

df['katakoryfo_0_ICE'] = ts.synoliko_katakoryfo(df["span"].shift(1).to_numpy(), df["height_diff"].shift(1).to_numpy(), df["tensions_0_ICE"].shift(1).to_numpy(), 2.5,
                                          df["span"].to_numpy(), df["height_diff"].to_numpy(), df["tensions_0_ICE"].to_numpy(), 2.5)

df['load_percentage_0_ICE'] = df['katakoryfo_0_ICE']*2*2.5/df['max_load_kg']*100
df['load_percentage_0_ICE'] = df['load_percentage_0_ICE'].round(2)

df['tensions_50_theoretical'] = 50.0

for i in range(len(df)):
    sign = df.iloc[i]["span_type"]
    df2=evaluate(diags[sign],[df.iloc[i]['span']])
    df.loc[i,'sag_50_theoretical']=float(df2.iloc[-1][str(df.iloc[i]['span'])])

# df['tensions_50_theoretical'] = ts.Th_from_sag(df['sag_50_theoretical'].to_numpy(),df['span'].to_numpy(),w)

df['tensions_50_theoretical'] = [
    ts.Th_from_sag(sag, span, w)
    for sag, span in zip(df['sag_50_theoretical'], df['span'])
]

df.loc[df["span_type"] == "BA350", "tensions_50_theoretical"] = 2585
df.loc[df["span_type"] == "BA500", "tensions_50_theoretical"] = 2654
df.loc[df["span_type"] == "term", "tensions_50_theoretical"] = 2585

df['katakoryfo_50_theoretical'] = ts.synoliko_katakoryfo(df["span"].shift(1).to_numpy(), df["height_diff"].shift(1).to_numpy(), df["tensions_50_theoretical"].shift(1).to_numpy(), w,
                                          df["span"].to_numpy(), df["height_diff"].to_numpy(), df["tensions_50_theoretical"].to_numpy(), w)

#monopleyro_left_vectorized = np.vectorize(ts.monopleyro_left)
#monopleyro_right_vectorized = np.vectorize(ts.monopleyro_right)

df["monopleyro_left_50_theoretical"] = ts.monopleyro_left(df["span"].shift(1).to_numpy(), df["height_diff"].shift(1).to_numpy(), df["tensions_50_theoretical"].shift(1).to_numpy(), w)
df["monopleyro_left_50_theoretical"] = df["monopleyro_left_50_theoretical"].shift(-1)
df["monopleyro_right_50_theoretical"] = ts.monopleyro_right(df["span"].to_numpy(), df["height_diff"].to_numpy(), df["tensions_50_theoretical"].to_numpy(), w)
df["monopleyro_right_50_theoretical"] = df["monopleyro_right_50_theoretical"].shift(1)

# df["monopleyro_left_50_theoretical"] = monopleyro_left_vectorized(df["span"].to_numpy(), df["height_diff"].to_numpy(), df["tensions_50_theoretical"].to_numpy(), w)
# df["monopleyro_right_50_theoretical"] = monopleyro_right_vectorized(df["span"].shift(-1).to_numpy(), df["height_diff"].shift(-1).to_numpy(), df["tensions_50_theoretical"].shift(-1).to_numpy(), w)

#print(df)

df["vari_Τ"] = 0.0
df["vari_Ζ"] = 0.0
#df["vari_1"] = 0.0
#df["vari_2"] = 0.0
#df["vari_3"] = 0.0
#df["vari_4"] = 0.0

for i in range(len(df)):
    if pd.isna(df.at[i, 'monopleyro_left_50_theoretical']):
        df.at[i, 'monopleyro_left_50_theoretical'] = 0.0

    if pd.isna(df.at[i, 'monopleyro_right_50_theoretical']):
        df.at[i, 'monopleyro_right_50_theoretical'] = 0.0

    sign = df.at[i, "type"]

    if sign[0] == "R" or sign[0] == "G" or sign[0] == "S":
        continue
    else:
        reduction1 = loads[sign] * 0.1 * 0.75
        reduction2 = loads[sign] * 0.1 * 65/80
        #reduction3 = loads[sing] * 0.1 * 0.46

        left  = float(df.at[i, 'monopleyro_left_50_theoretical'])
        right = float(df.at[i, 'monopleyro_right_50_theoretical'])

        df.at[i,'vari_Τ'] = max(0,(left + right - reduction1) * 2 * 2.623)
        df.at[i,'vari_Ζ'] = max(0,(left + right - reduction2) * 2 * 2.623)

        # df.at[i,'vari_1'] = max(0,(left + right - reduction1) * 2 * 1.823)
        # df.at[i,'vari_2'] = max(0,(left + right - reduction2) * 2 * 1.823)
        # df.at[i,'vari_3'] = max(0,(left + right - reduction1) * 2 * 2.623)
        # df.at[i,'vari_4'] = max(0,(left + right - reduction2) * 2 * 2.623)

###############################################################
###
###  λογική για θετικό μονόπλευρο
###
###############################################################

# df["apostasi_left"] = ts.distance_lowest_point_l(df["span"].shift(1).to_numpy(), df["height_diff"].shift(1).to_numpy(), df["tensions_50_theoretical"].shift(1).to_numpy(), w)
# df["apostasi_right"] = ts.distance_lowest_point_r(df["span"].to_numpy(), df["height_diff"].to_numpy(), df["tensions_50_theoretical"].to_numpy(), w)

# df["thetika_vari_Τ"] = 0.0
# df["thetika_vari_Ζ"] = 0.0

# for i in range(len(df)):

#     sign = df.at[i, "type"]
    
#     left = df.at[i, "apostasi_left"]
#     right = df.at[i, "apostasi_right"]

#     if sign[0] == "R" or sign[0] == "G" or sign[0] == "S" or left<0 or right<0 or max(left,right)<loads[sign]/2:
#         continue
#     else:
#         reduction1 = loads[sign] * 0.75 / 2
#         reduction2 = loads[sign] * 65/80 / 2 

#         thetiko = max(left, right)

#         df.at[i,'thetika_vari_Τ'] = (thetiko - reduction1) * 2 * 2.623
#         df.at[i,'thetika_vari_Ζ'] = (thetiko - reduction2) * 2 * 2.623  
 


#### df.to_excel("larisa2_processed.xlsx", index=False) 

df.to_excel(HERE / "outputs" / "larisa1_v2_processed.xlsx", index=False)