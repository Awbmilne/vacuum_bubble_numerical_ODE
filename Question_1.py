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

import csv
from os import times
import numpy
import timeit
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sympy import Matrix, cos, symbols, Eq
from sympy.matrices import Matrix

from Solvy_boi import Solvy_boi

out_path = Path('./out/Question_1')

if __name__ == "__main__":
    # CONFIGURATION ─────────────────────────────────────────────────────────────────── #
    r_0 = 0.002
    rho = 1000
    p_0 = 100981
    lmda_sqr = (3 * p_0) / (rho * r_0**2)
    T_star = 0.0003609935174 
    delta_t = 0.000001

    # Create the necessary symbols
    P, R, t = symbols("P R t")

    # Create the ODE object
    system = Solvy_boi(
        [P, R], # List of variables, The slopes of which are the LHS of the below equations
        [-(3/2) * (p_0/(rho*r_0)) + lmda_sqr*r_0 - lmda_sqr*R, P], # List of functions, RHS of system
        [lambda t: None, lambda t : 0.001 * cos(8702.629 * t) + 0.001]
    )
    state_0 = Matrix([0,0.002]) # Initial state of system

    # CALCULATIONS ──────────────────────────────────────────────────────────────────── #
    # Run the computation using each method and collect data
    data = {}
    time = {}
    methods = [["Eulers",     Solvy_boi.e_eul],
               ["RK2",        Solvy_boi.e_rk2],
               ["RK4",        Solvy_boi.e_rk4],
               ["Analytical", Solvy_boi.anl]]
    for method in methods:
        start = timeit.default_timer()
        data[method[0]] = system.run_solution(method[1], state_0, delta_t, T_star)
        time[method[0]] = timeit.default_timer() - start
    data.update((label, pd.DataFrame(set)) for label, set in data.items()) # Convert data sets to dataframes

    # Add column names for data in each data frame (prettify)
    plot_symbols = ['t'] + [repr(sym) for sym in system.symbols] # List of symbols (prepend 't')
    for _, set in data.items():
        set.rename(columns=dict(enumerate(plot_symbols, start=0)), inplace=True) # Name colums of the data sets

    # DATA OUTPUT ───────────────────────────────────────────────────────────────────── #
    # Print the Data Sets for posterity
    for label, set in data.items():
        print(f"\n ── Data for system solved using {label} method ──")
        print(f"Solution time: {time[label]}")
        print(numpy.shape(set), type(set), set, sep='\n') # Print to stdout
        set.to_csv(out_path / f"{label}_data") # Save CSV file to 'out' folder

    with open(out_path / f"solve_times", 'w') as file:
        labels = [label for label,_ in time.items()]
        writer = csv.DictWriter(file, labels)
        # writer.writerow(labels)
        writer.writerow(time)

    # Create a plot for each data set
    for label, set in data.items():
        plt.plot(set[repr(t)], set[repr(R)]) # Plot each line with its symbol
        plt.title(f"Radius vs Time - {label} Method - Δt={delta_t}")
        plt.xlabel("Time")
        plt.ylabel("Radius")
        plt.savefig(out_path / f"{label}_graph.png")
        plt.clf()
    
    # Create combined plot for all solutions
    for label, set in data.items():
        plt.plot(set[repr(t)], set[repr(R)], label=label)
    plt.title(f"Radius vs Time - Methods Compared - Δt={delta_t}")
    plt.xlabel("Time")
    plt.ylabel("Radius")
    plt.legend()
    plt.savefig(out_path / f"combined_graph.png")
    plt.show()
