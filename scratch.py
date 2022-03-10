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

import sys
import copy
import math
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import E
from sympy import Matrix, sin, cos, symbols
from sympy.matrices import Matrix

## BOOKMARK: ODE Solver Class
class Solvy_boi:
    """ 
    ODE numerical solution class.
    This Class takes a system of first order ODEs and provides a number of
    methods available for solving the system numerically.
    """
    def __init__(self, symbols, functions):
        # Store the ODE system
        self.symbols = symbols
        self.functions = functions
    
    def e_eul(self, state, dt):
        # Increment the positions using Explicit Euler method
        v_subs_dict = list(zip(self.symbols, state)) # Dictionary for value substitution
        slopes = Matrix([f.subs(v_subs_dict) for f in self.functions]) # Solve for slopes
        state = state + dt * slopes # Apply slopes to state
        return state
    
    def e_rk2(self, state, dt):
        # Increment the state using Runge-Kutta (RK2) method
        state_0 = copy.copy(state)
        # Solve for k1
        v_subs_dict = list(zip(self.symbols, state))
        k1 = Matrix([f.subs(v_subs_dict) for f in self.functions])
        # Solve for k2 using k1
        state = state_0 + 0.5 * dt * k1
        v_subs_dict = list(zip(self.symbols, state))
        k2 = Matrix([f.subs(v_subs_dict) for f in self.functions])
        # Apply the k2 slope to the function
        state = state_0 + dt * k2
        return state

    def e_rk4(self, state, dt):
        # Increment the state using Runge-Kutta (RK4) method
        state_0 = copy.copy(state)
        # Solve for k1
        v_subs_dict = list(zip(self.symbols, state))
        k1 = Matrix([f.subs(v_subs_dict) for f in self.functions])
        # Solve for k2 using k1
        state = state_0 + 0.5 * dt * k1
        v_subs_dict = list(zip(self.symbols, state))
        k2 = Matrix([f.subs(v_subs_dict) for f in self.functions])
        # Solve for k3 using k2
        state = state_0 + 0.5 * dt * k2
        v_subs_dict = list(zip(self.symbols, state))
        k3 = Matrix([f.subs(v_subs_dict) for f in self.functions])
        # Solve for k4 using k3
        state = state_0 + dt * k3
        v_subs_dict = list(zip(self.symbols, state))
        k4 = Matrix([f.subs(v_subs_dict) for f in self.functions])
        # Apply the k slopes to the function
        state = state_0 + dt * (1.0/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return state
    
    def run_solution(self, method, state_0, d_t ,t):
        state = state_0 # Not necessary, just cleanliness
        data_set = [[0] + [v for v in state_0]] # Store the 0 initial data point
        for i in range(int(t / d_t)): # For every incremental step
            state = method(self, state, d_t) # Update the state using the specified method
            data_set.append([d_t*(i+1)] + [v for v in state]) # Append the data to the array
        return data_set

## BOOKMARK: Main Script
if __name__ == "__main__":
    r_0 = 0.002
    rho = 1000
    p_0 = 100981
    lmda_sqr = (3 * p_0) / (rho * r_0**2)
    T_star = 0.0003609935174 
    delta_t = 0.00000001

    # Create the necessary symbols
    P, R = symbols("P R")

    # Create the ODE object
    system = Solvy_boi(
        [P, R], # List of variables, The slopes of which are the LHS of the below equations
        [-(3/2) * (p_0/(rho*r_0)) + lmda_sqr*r_0 - lmda_sqr*R, P] # List of functions, RHS of system
    )
    state_0 = Matrix([0,0.002]) # Initial state of system

    # Run the computation using each method
    # data = system.run_solution(Solvy_boi.e_eul, state_0, delta_t, T_star)
    # data = system.run_solution(Solvy_boi.e_rk2, state_0, delta_t, T_star)
    data = system.run_solution(Solvy_boi.e_rk4, state_0, delta_t, T_star)

    # Create prettified data frame
    df = pd.DataFrame(data) # Create a pandas data frame from the data array
    # pd.set_option("display.max_rows", None)
    plot_symbols = ['t'] + [repr(s) for s in system.symbols] # Create a list of symbols for the dataframe (Add 't')
    df.rename(columns=dict(enumerate(plot_symbols, start=0)), inplace=True) # Name each part of the data frame (clean output)
    df['A'] = [r_0] + [0.001 * cos(8702.629 * delta_t * (i+1)) + 0.001 for i in range(int(T_star / delta_t))] # Add Analytical solution
    print(numpy.shape(df), type(df), df, sep='\n') # Output the data frame

    # Create a plot from the data
    for symbol in df.columns[2:4]:
        plt.plot(df[plot_symbols[0]], df[symbol], label='Line '+symbol) # Plot each line with its symbol

    plt.legend() # Enable legend
    plt.show() # Show plot
