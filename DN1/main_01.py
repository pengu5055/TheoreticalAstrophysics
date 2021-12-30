import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, quad
import cmasher as cmr
from src_methods import *
import matplotlib.colors as color


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


def zero_filter(array):
    """Input array must be ndarray"""
    return array[array > 0]


def xi_to_r(xi, xi1):
    """Convert from dimensionless xi to dimensionless r/R"""
    return xi/xi1


def rho_c(xi1, dtheta):
    M = -4


def density_norm(theta, n):
    """Returns density normalised to central density \rho/\rho_c"""
    return theta**n


def pressure_norm(theta, n):
    """Returns P= K\rho_c^{1 + 1/n}\theta^{n+1} but with constants set to 1"""
    return theta**(n+1)/theta[0]**(n+1)


def temperature_norm(theta, n):
    r"""Returns T/T_c"""
    return theta/theta[0]


def mass_norm(theta, xi, n, xi1):
    theta = zero_filter(theta)
    dim = len(theta)
    output = []
    M = np.sum([xi_to_r(xi[j], xi1)**2 * theta[j]**n for j in range(dim)])  # Whole mass without constants

    for i in range(dim):
        m = np.sum([xi_to_r(xi[j], xi1)**2 * theta[j]**n for j in range(i)])
        output.append(m/M)
        print("Step: {}".format(i))

    return xi[:dim], np.array(output)


# Test plot
# n = 1.5
# theta_init = [1, 0]
# xi_range = np.linspace(0.1, 10, 100)

# sol = np.column_stack(odeint(lane_emden, theta_init, xi_range))
# sol = vec_midpoint(lane_emden, theta_init, xi_range)
# print(sol)
# plt.plot(xi_range, sol[0])
# plt.show()

# Plot different solutions
# n_list = [0, 0.1, 0.5, 1, 1.5, 3, 5]
# theta_init = [1, 0]
# xi_range = np.linspace(0.01, 100, 50000)
# colors = cmr.take_cmap_colors("cmr.cosmic", 7, cmap_range=(0.15, 1), return_fmt="hex")  # For nicer colors
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
# plt.axhline(alpha=1, ls=":", c="#adadad")
# plt.ylim(-0.4, 1.1)
# plt.show()

# Plot derivatives
# n_list = [0, 0.1, 0.5, 1, 1.5, 3, 5]
# theta_init = [1, 0]
# xi_range = np.linspace(0.01, 10, 1000000)
# colors = cmr.take_cmap_colors("cmr.cosmic", 7, cmap_range=(0.15, 1), return_fmt="hex")


# for index, i in enumerate(n_list):
#     n = i
#     sol = np.column_stack(odeint(lane_emden, theta_init, xi_range))
#     plt.plot(xi_range, sol[1], label="n = {}".format(i), c=colors[index])

# plt.title("Odvod Lane-Emdenove enačbe")
# plt.xlabel(r"$\xi$")
# plt.ylabel(r"$\theta$")
# plt.legend()
# plt.axhline(alpha=1, ls=":", c="#adadad")
# plt.ylim(-0.8, 0.2)
# plt.show()

# Continuous version
# div = 200
# n_list = np.linspace(0, 5, div)
# theta_init = [1, 0]
# xi_range = np.linspace(0.01, 10, 100)
# winter = plt.cm.get_cmap("winter", div)
#
# for index, i in enumerate(n_list):
#     n = i
#     sol = np.column_stack(odeint(lane_emden, theta_init, xi_range))
#     plt.plot(xi_range, sol[0], c=winter(index))
#
# plt.title("Rešitve Lane-Emdenove enačbe")
# plt.xlabel(r"$\xi$")
# plt.ylabel(r"$\theta$")
# plt.legend()
# plt.axhline(alpha=1, ls=":", c="#adadad")
# plt.ylim(-0.4, 1.1)
# plt.colorbar(plt.cm.ScalarMappable(cmap=winter, norm=color.Normalize(vmin=0, vmax=5)),
#              label="Politropni indeks")
# plt.show()

# Density, Mass, Temperature, Pressure plots n = 1.5 (monoatomic gas)
dims = 5000
c1, c2 = cmr.take_cmap_colors("cmr.cosmic", 2, cmap_range=(0.5, 1), return_fmt="hex")

n = 1.5
xi1 = 3.64
theta_init = [1, 0]
xi_range = np.linspace(0.01, xi1, dims)
theta, dtheta = np.column_stack(odeint(lane_emden, theta_init, xi_range))
r = xi_to_r(xi_range, xi1)
# Density, Mass, Temperature, Pressure plots n = 3 (relativistic degenerate gas)
n = 3
xi1_2 = 6.89
xi_range2 = np.linspace(0.01, xi1_2, dims)
theta2, dtheta2 = np.column_stack(odeint(lane_emden, theta_init, xi_range2))
r2 = xi_to_r(xi_range2, xi1_2)


# Density plot
rho = density_norm(theta, 1.5)
rho2 = density_norm(theta2, 3)

plt.plot(r, rho, c=c1, label=r"$n = 1.5$")
plt.plot(r2, rho2, c=c2, label=r"$n = 3$")

plt.title("Gostota")
plt.xlabel(r"$r/R$")
plt.ylabel(r"$\rho/\rho_c$")
plt.legend()
plt.show()

# Mass plot
x, m = mass_norm(theta, xi_range, 1.5, xi1)
x2, m2 = mass_norm(theta2, xi_range2, 3, xi1_2)

plt.plot(xi_to_r(x, xi1), m, c=c1, label=r"$n = 1.5$")
plt.plot(xi_to_r(x2, xi1_2), m2, c=c2, label=r"$n = 3$")

plt.title("Masa")
plt.xlabel(r"$r/R$")
plt.ylabel(r"$m/M$")
plt.legend()
plt.show()

# Pressure plot
p = pressure_norm(theta, 1.5)
p2 = pressure_norm(theta2, 3)

plt.plot(r, p, c=c1, label=r"$n = 1.5$")
plt.plot(r2, p2, c=c2, label=r"$n = 3$")

plt.title("Tlak")
plt.xlabel(r"$r/R$")
plt.ylabel(r"$p/p_c$")
plt.legend()
plt.show()

# Temperature plot
T = temperature_norm(theta, 1.5)
T2 = temperature_norm(theta2, 3)

plt.plot(r, T, c=c1, label=r"$n = 1.5$")
plt.plot(r2, T2, c=c2, label=r"$n = 3$")

plt.title("Temperatura")
plt.xlabel(r"$r/R$")
plt.ylabel(r"$T/T_c$")
plt.legend()
plt.show()
