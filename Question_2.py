__author__ = "Austin W. Milne"
__credits__ = ["Austin W. Milne", "Japmeet Brar", "Joshua Selvanayagam", "Kevin Chu"]
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
import csv
import numpy
import timeit
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sympy import Matrix, symbols
from sympy.matrices import Matrix

from Solvy_boi import Solvy_boi

def run_q2(delta_t, T_star, slope_lim, root, r_0=100, show_output=False):
    # CONFIGURATION ------------------------------------------------------------------- #
    rho = 996
    mu = 0.798e-3
    sigma = 0.072
    p_0 = 10e5 + 1000*9.81*100
    out_p = root / f'dt_{delta_t}'
    
    # Ensure output directory exists
    if not os.path.exists(out_p):
        os.makedirs(out_p)

    # Create the necessary symbols
    P, R, t = symbols("P R t")

    # Create the ODE object
    system = Solvy_boi(
        R, # The actual output variable
        [P, R], # List of variables, The slopes of which are the LHS of the below equations
        [(-p_0 - 4*mu*(P/R)-2*(sigma/R)-(3/2)*(rho*P**2))/(rho * R), P], # List of functions, RHS of system
        slope_lim,
        [lambda t: None, lambda t : None]
    )
    state_0 = Matrix([0,r_0]) # Initial state of system

    # CALCULATIONS -------------------------------------------------------------------- #
    # Run the computation using each method and collect data
    data = {}
    time = {}
    methods = [["Eulers",     Solvy_boi.e_eul],
               ["RK2",        Solvy_boi.e_rk2],
               ["RK4",        Solvy_boi.e_rk4],]
    for method in methods:
        start = timeit.default_timer()
        data[method[0]] = system.run_solution(method[1], state_0, delta_t, T_star)
        time[method[0]] = timeit.default_timer() - start
    data.update((label, pd.DataFrame(set)) for label, set in data.items()) # Convert data sets to dataframes

    # Add column names for data in each data frame (prettify)
    plot_symbols = ['t'] + [repr(sym) for sym in system.symbols] # List of symbols (prepend 't')
    for _, set in data.items():
        set.rename(columns=dict(enumerate(plot_symbols, start=0)), inplace=True) # Name colums of the data sets

    # DATA OUTPUT --------------------------------------------------------------------- #
    # Print the Data Sets for posterity
    for label, set in data.items():
        if show_output: print(f"\n -- Data for system solved using {label} method --")
        if show_output: print(numpy.shape(set), type(set), set, sep='\n') # Print to stdout
        print(f"Solution time ({label}): {time[label]}")
        set.to_csv(out_p / f"{label}_data.csv") # Save CSV file to 'out' folder

    with open(out_p / f"solve_times", 'w') as file:
        labels = [label for label,_ in time.items()]
        writer = csv.DictWriter(file, labels)
        # writer.writerow(labels)
        writer.writerow(time)

    # Create a plot for each data set
    for label, set in data.items():
        plt.plot(set[repr(t)], set[repr(R)]) # Plot each line with its symbol
        # plt.title(f"Radius vs Time - {label} Method - Δt={delta_t}")
        plt.xlabel("Time")
        plt.ylabel("Radius")
        plt.savefig(out_p / f"{label}_graph.png", bbox_inches = 'tight')
        plt.clf()
    
    # Create combined plot for all solutions
    for label, set in data.items():
        plt.plot(set[repr(t)], set[repr(R)], label=label)
    # plt.title(f"Radius vs Time - Methods Compared - Δt={delta_t}")
    plt.xlabel("Time")
    plt.ylabel("Radius")
    plt.legend()
    plt.savefig(out_p / f"combined_graph.png", bbox_inches = 'tight')
    if show_output: plt.show()
    plt.clf()


if __name__ == '__main__':
    delta_t = 0.0001
    T_star = 15
    slope_limit = 10e6
    root = Path(f'./out/Question_2')
    run_q2(delta_t, T_star, slope_limit, root, show_output=True)
