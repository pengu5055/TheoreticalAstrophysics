import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import cmasher as cmr
from src_methods import *


def lane_emden(theta, xi):
    """
    Lane-Emdden equation for a simple polytopic model.
    theta'' = -[theta^n + 2/xi thata']

    INPUT:
    theta = 2D vector of theta and theta'
    xi = scalar
    OUTPUT:
    Returns second derivative
    """
    return theta[1], -(theta[0]**n + 2/xi*theta[1])


def lane_emden2(theta, xi):
    return -theta[1]/xi**2, theta[0]**n * xi**2


# Test
n = 1.5
theta_init = [1, 0]
xi_range = np.linspace(0.1, 10, 100)

# sol = np.column_stack(odeint(lane_emden, theta_init, xi_range))
sol = vec_midpoint(lane_emden2, theta_init, xi_range)
print(sol)
plt.plot(xi_range, sol[0])
plt.show()

x, time = ([-0.019697923980261402, -0.20519560436313294], 3.6)
value = lane_emden(x, time)
print(value)
