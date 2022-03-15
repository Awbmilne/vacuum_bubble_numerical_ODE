__author__ = "Austin W. Milne"
__credits__ = ["Austin W. Milne", ]
__email__ = "awbmilne@uwaterloo.ca"
__version__ = "1.0"
__date__ = "March 8, 2022"

"""
This code was written for Project #1 of ME303 "Advanced Engineering Mathmatics" in the Winter 2022 term.
The goal is to numerically solve an ODE relating to the occilation of a collapsing vacuum bubble in
water. There are a number of solution methods implemented and compared. In some cases, these solutions
are also compared to the analytical solution.
"""

import os
import re
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from Question_1 import run_q1
from Question_2 import run_q2

# Q1 AUTORUN CONFIGURATION -------------------------------------------------------- #
Q1_dt_variance_root = Path(f'./out/dt_variance/Question_1')
Q1_delta_t_max = 0.00001
Q1_delta_t_steps = 10
Q1_delta_t_factor = 2

Q1_T_star = 0.0003609935174 
Q1_slope_lim = 10e5

Q1_error_ref_time = 0.0003

# Logarithmic Δt set
Q1_dt_set = [Q1_delta_t_max / (Q1_delta_t_factor**i) for i in range(Q1_delta_t_steps+1)]

# Q2 AUTORUN CONFIGURATION -------------------------------------------------------- #
Q2_dt_variance_root = Path(f'./out/dt_variance/Question_2')
Q2_delta_t_max = 0.01
Q2_delta_t_steps = 10
Q2_delta_t_factor = 2

Q2_T_star = 15
Q2_slope_lim = 10e5

Q_2_error_time = 9

# Logarithmic Δt set
Q2_dt_set = [Q2_delta_t_max / (Q2_delta_t_factor**i) for i in range(Q2_delta_t_steps+1)]

Q2_size_variance_root = Path(f'./out/size_variance/Question_2')
Q2_size_set = [0.1, 1, 10, 100]

if __name__ == '__main__':
    # AUTORUN ------------------------------------------------------------------------- #
    # Debugging output of Δt sets
    # print(f"Q1 dt set:\n{Q1_dt_set}")
    # print(f"Q2 dt set:\n{Q2_dt_set}")
    
    # # Run the Q1 set
    # for i, dt in enumerate(Q1_dt_set, start=1):
    #     print(f"\nRunning ({i}/{len(Q1_dt_set)}) Q1 with dt = {dt}")
    #     run_q1(dt, Q1_T_star, Q1_slope_lim, Q1_dt_variance_root)

    # # Run the Q2 set
    # for i, dt in enumerate(Q2_dt_set, start=1):
    #     print(f"\nRunning ({i}/{len(Q2_dt_set)}) Q2 with dt = {dt}")
    #     run_q2(dt, Q2_T_star, Q2_slope_lim, Q2_dt_variance_root)


    # ERROR DETERMINATION AND GRAPHING ------------------------------------------------ #
    # Determine error for each question
    error_root = Path('./out/Error')
    questions = ['Question_1', 'Question_2']
    roots = [Q1_dt_variance_root, Q2_dt_variance_root]
    reference_times = [0.00034, 14.0]
    reference_methods = ['Analytical', 'RK4']
    methods = [["Eulers", "RK2", "RK4"],
            ["Eulers", "RK2", "RK4"]]
    for q, root, time, method, methods in zip(questions, roots, reference_times, reference_methods, methods):
        # Create a sorted list of the Δt values
        dts = []
        for dir in os.listdir(root):
            m = re.search(r'(?<=dt_).*', dir)
            dts.append(float(m.group(0)))
        dts.sort(reverse=True)

        # Set the reference point for error calculation
        ref_pnt = 0.0
        with open(root / f'dt_{dts[-1]}' / f'{method}_data.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[1] == str(time):
                    ref_pnt = float(row[3])
                    break
        
        # Collect list of error data
        error_list = []
        for dt in dts:
            error_frame = [dt]
            for method in methods:
                with open(root / f'dt_{dt}' / f'{method}_data.csv', newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        if row[1] == str(time):
                            error_frame.append(abs(float(ref_pnt) - float(row[3])))
            error_list.append(error_frame)
        
        # Create labeled dataframe for easier data manipulation
        df = pd.DataFrame(error_list)
        columns = ['dt'] + methods
        df.rename(columns=dict(enumerate(columns, start=0)), inplace=True)

        # Create combined plot of data values
        out_file = error_root / q / f"error.png"
        if not os.path.isdir(Path(out_file).parent):
            os.makedirs(Path(out_file).parent)
        for method in methods:
            plt.plot(df['dt'], df[method], label=method) # Plot the single method
        plt.title(f"Method error vs Δt - {q.replace('_',' ')}")
        plt.xlabel("Δt [sec]")
        plt.ylabel("error [m]")
        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.savefig(error_root / q / f"error.png")
        plt.clf()


    # VARIED BUBBLE DIAMETER ---------------------------------------------------------- #
    
    # Run the Q2 set
    for i, size in enumerate(Q2_size_set, start=1):
        dt = min(Q2_dt_set)
        print(f"\nRunning ({i}/{len(Q1_dt_set)}) Q2 with size = {size}")
        run_q2(dt, Q2_T_star, Q2_slope_lim, Q2_size_variance_root / str(size), r_0=size)