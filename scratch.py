
import copy
from sympy import Function, Symbol, Matrix, sin, cos, symbols, diff, Derivative, Array
from sympy.solvers import solve
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt

## TODO: Eliminate extraneous slope calculations and storage

class Solvy_boi:
    def __init__(self, symbols, functions):
        self.symbols = symbols
        self.functions = functions
        self.state_size = len(symbols) * 2
    
    def e_eul(self, pos, slope, dt): ## NOTE: Probably dont need to store and re-calculate the slop the second time around
        # Increment the positions using Explicit Euler method
        value_subs_array = list(zip(self.symbols, pos))
        slopes = Matrix([f.subs(value_subs_array) for f in self.functions])
        pos = pos + dt * slopes
        
        # Increment the slopes
        new_value_subs_array = list(zip(self.symbols, pos))
        slope = Matrix([f.subs(new_value_subs_array) for f in self.functions])
        return pos, slope
    
    def e_rk2(self, pos, slope, dt):
        # Increment the positions using Runge-Kutta (RK4) method
        pos_i = copy.copy(pos)
        slope_i = copy.copy(slope) ## NOTE: Probably dont need
        # Solve for k1
        value_subs_array = list(zip(self.symbols, pos))
        slopes = Matrix([f.subs(value_subs_array) for f in self.functions])
        # Solve for k2 using k1
        pos = pos_i + 0.5 * dt * slopes
        value_subs_array = list(zip(self.symbols, pos))
        slopes = Matrix([f.subs(value_subs_array) for f in self.functions])
        # Apply the k slopes to the function
        pos = pos_i + dt * slopes
        new_value_subs_array = list(zip(self.symbols, pos)) ## NOTE: Probably dont need
        slope = Matrix([f.subs(new_value_subs_array) for f in self.functions]) ## NOTE: Probably dont need
        return pos, slope

    def e_rk4(self, pos, slope, dt):
        # Increment the positions using Runge-Kutta (RK4) method
        pos_i = copy.copy(pos)
        slope_i = copy.copy(slope) ## NOTE: Probably dont need
        rk_slopes = []
        # Solve for k1
        value_subs_array = list(zip(self.symbols, pos))
        slopes = Matrix([f.subs(value_subs_array) for f in self.functions])
        rk_slopes.append(slopes)
        # Solve for k2 using k1
        pos = pos_i + 0.5 * dt * slopes
        value_subs_array = list(zip(self.symbols, pos))
        slopes = Matrix([f.subs(value_subs_array) for f in self.functions])
        rk_slopes.append(slopes)
        # Solve for k3 using k2
        pos = pos_i + 0.5 * dt * slopes
        value_subs_array = list(zip(self.symbols, pos))
        slopes = Matrix([f.subs(value_subs_array) for f in self.functions])
        rk_slopes.append(slopes)
        # Solve for k4 using k3
        pos = pos_i + dt * slopes
        value_subs_array = list(zip(self.symbols, pos))
        slopes = Matrix([f.subs(value_subs_array) for f in self.functions])
        rk_slopes.append(slopes)
        # Apply the k slopes to the function
        pos = pos_i + dt * (1.0/6.0) * (rk_slopes[0] + 2*rk_slopes[1] + 2*rk_slopes[2] + rk_slopes[3])
        new_value_subs_array = list(zip(self.symbols, pos)) ## NOTE: Probably dont need
        slope = Matrix([f.subs(new_value_subs_array) for f in self.functions]) ## NOTE: Probably dont need
        return pos, slope
    
    def run_solution(self, method, pos_0, slope_0, d_t ,t):
        pos = pos_0
        slope = slope_0
        data_set = [(0*d_t, pos_0, slope_0)]
        for i in range(int(t / d_t)):
            pos, slope = method(self, pos, slope, d_t) ## NOTE: Slope probably not necessary
            data_set.append((i+1, pos, slope))
        return data_set


a,b,c = symbols("a b c")

my_system = Solvy_boi(
    [a, b, c],
    [b**2 + cos(b), a, a * b**2]
)


if __name__ == "__main__":
    print(my_system)
    pos = Matrix([0,0,0])
    slope = Matrix([0,0,0])

    data = my_system.run_solution(Solvy_boi.e_eul, pos, slope, 0.0001, 1)
    print(data[-1])
    data = my_system.run_solution(Solvy_boi.e_rk2, pos, slope, 0.0001, 1)
    print(data[-1])
    data = my_system.run_solution(Solvy_boi.e_rk4, pos, slope, 0.0001, 1)
    print(data[-1])
