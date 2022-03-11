import copy
import math
import numpy as np
from sympy import E, Matrix
from sympy.matrices import Matrix

s_lim = 10e5

## BOOKMARK: ODE Solver Class
class Solvy_boi:
    """ 
    ODE numerical solution class.
    This Class takes a system of first order ODEs and provides a number of
    methods available for solving the system numerically.
    """
    def __init__(self, symbol, symbols, functions, analytical):
        # Store the ODE system
        self.symbol = symbol
        self.symbols = symbols
        self.functions = functions
        self.analytical = analytical
    
    def e_eul(self, state, dt, t):
        # Increment the positions using Explicit Euler method
        v_subs_dict = list(zip(self.symbols, state)) # Dictionary for value substitution
        slopes = Matrix([np.clip(f.subs(v_subs_dict),-s_lim,s_lim) for f in self.functions]) # Solve for slopes
        new_state = state + dt * slopes # Apply slopes to state
        # Rebound when the radius goes below 0
        i = self.symbols.index(self.symbol)
        if math.copysign(1, state[i]) != math.copysign(1, new_state[i]):
            if (state[i] != 0) and (new_state[i] != 0):
                state[i] = -state[i]
                return -state
        return new_state
    
    def e_rk2(self, state, dt, t):
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

    def e_rk4(self, state, dt, t):
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
    
    def anl(self, state, dt, t):
        # Solve given lambda for the specified t
        return [lmb(t) for lmb in self.analytical]
    
    def run_solution(self, method, state_0, d_t ,t):
        state = state_0 # Not necessary, just cleanliness
        data_set = [[0] + [v for v in state_0]] # Store the 0 initial data point
        for i in range(int(t / d_t)): # For every incremental step
            time = d_t*(i+1)
            state = method(self, state, d_t, time) # Update the state using the specified method
            data_set.append([d_t*(i+1)] + [v for v in state]) # Append the data to the array
        return data_set