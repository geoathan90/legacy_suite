"""
    A script to compare the results of the new and old implementations of 
    the function for problem 31185. 
    It evaluates both implementations over  a range of x values, calculates 
    the average of the two results, and then computes the relative difference
    between them. Finally, it identifies the maximum absolute difference and 
    the corresponding x value where this occurs.
"""
from .eval import evaluate
import numpy as np
#import matplotlib.pyplot as plt

x_max = 500
x_min = 150

points = (x_max - x_min)*100+1
x_range = np.linspace(x_min, x_max, points)

df_new = evaluate("31185", x_range)
#print(df_new)

df_old = evaluate("31185_old", x_range)
#print(df_old)

df_avg = (df_new + df_old) / 2

df = (df_new - df_old)/df_avg
#print(df)


abs_df = df.abs()

max_abs_diff = abs_df.max(axis=1)
max_abs_diff_x = abs_df.idxmax(axis=1)

print("Max absolute difference:", max_abs_diff.max())
print("At x =", max_abs_diff_x[max_abs_diff.idxmax()])