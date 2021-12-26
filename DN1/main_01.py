import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import cmasher as cmr


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


# Test
n = 1.5
theta_init = [1, 0]
xi_range = np.linspace(0.1, 10, 500)

sol = np.column_stack(odeint(lane_emden, theta_init, xi_range))
print(sol)
plt.plot(xi_range, sol[0])
plt.show()

# Plot different solutions
# n_list = [0, 0.1, 0.5, 1, 1.5, 3, 5]
# theta_init = [1, 0]
# xi_range = np.linspace(0.01, 10, 1000)
# colors = cmr.take_cmap_colors("cmr.cosmic", 7, return_fmt="hex")
#
#
# for index, i in enumerate(n_list):
#     n = i
#     sol = np.column_stack(odeint(lane_emden, theta_init, xi_range))
#     plt.plot(xi_range, sol[0], label="n = {}".format(i), c=colors[index])
#
# plt.title("Rešitve Lane-Emdenove enačbe")
# plt.xlabel(r"$\xi$")
# plt.ylabel(r"$\theta$")
# plt.legend()
# plt.ylim(-1.2, 1.2)
# plt.show()

