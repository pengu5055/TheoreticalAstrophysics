"""Additional numerical methods and functions used in DN1 to prevent clutter"""
import numpy as np


def vec_rk4(f, x0, t):
    h = t[1] - t[0]
    x = x0
    output = []
    for time in t:
        k1 = h * np.asarray(f(x, time))
        k2 = h * np.asarray(f(x + 1/2 * k1, time + h/2))
        k3 = h * np.asarray(f(x + 1/2 * k2, time + h/2))
        k4 = h * np.asarray(f(x + k3, time + h))
        x += 1/6 * (k1 + 2*k2 + 2*k3 + k4)
        output.append(x)

    return np.column_stack(output)


def vec_midpoint(f, x0, t):
    h = t[1] - t[0]
    x = x0
    output = []
    c = 0  # Debug snippet
    for time in t:
        row = []
        f_vec = f(x, time)
        n = np.shape(f_vec)[0]
        k1 = [h*f_vec[i]/2 for i in range(n)]
        f_vec2 = f(x + k1, time + h/2)
        for i in range(n):
            x[i] += h * f_vec2[i]
            row.append(x[i])

        output.append(row)
        c += 1

    return np.column_stack(np.array(output))