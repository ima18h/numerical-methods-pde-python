import numpy as np
from util import cfl_stability as stability
from numba import jit

def initialize(nx, nt, dx, dt, c, u0):
    stable = stability(c, dx, dt)
    if True:
        matrix = np.zeros([nt, nx])

        for i in range(1, nx-1):
            matrix[0, i] = u0(i * dx)

        result_matrix = simulate(nx, nt, dx, dt, c, matrix)
        return result_matrix

    else:
        exit()


# This uses the upwind scheme
def simulate(nx, nt, dx, dt, c, u):
    for n in range(1, nt):  # loop for values of n from 0 to nt, so it will run nt times
        for j in range(1, nx):
            u[n, j] = u[n - 1, j] - c * (dt / dx) * (u[n - 1, j] - u[n - 1, j - 1])
    return u


# For analytical solution
def initialize_ana(nx, nt, dx, dt, c, u0):
        matrix_ana = np.zeros([nt, nx])

        for i in range(1, nx):
            matrix_ana[0, i] = u0(i * dx)

        for n in range(1, nt):  # loop for values of n from 0 to nt, so it will run nt times
            for i in range(1, nx):
                matrix_ana[n, i] = u0((i * dx) - c*(n*dt))

        return matrix_ana

def initialize_2d(nx, ny, nt, dx, dy, dt, c, u0, x_0, y_0):
    stable = stability(c, dx, dt)
    matrix = np.zeros((nt, ny, nx))
    c_x = (c * dt) / (2 * dx)
    c_y = (c * dt) / (2 * dy)

    for j in range(1, ny-1):
        for i in range(1, nx-1):
            matrix[0, j, i] = u0(x_0 + i * dx, y_0 + j * dy)

    # result_matrix = simulate_2d(nx, ny, nt, dx, dt, c_x, c_y, matrix)
    return simulate_2d(nx, ny, nt, c_x, c_y, matrix)

@jit
def simulate_2d(nx, ny, nt, c_x, c_y, u):
    for n in range(nt-1):  # loop for values of n from 0 to nt, so it will run nt times
        for j in range(1, nx - 1):
            for i in range(1, ny - 1):
                u[n + 1, j, i] = u[n, j, i] - c_x*(u[n, j, i+1] - u[n, j, i-1]) - c_y*(u[n, j+1, i] - u[n, j-1, i])
    return u

def initialize_2d_scheme2(nx, ny, nt, dx, dy, dt, c, u0, x_0, y_0):
    matrix = np.zeros((nt, ny, nx))
    c_x = (c * dt) / dx
    c_y = (c * dt) / dy

    for j in range(1, ny-1):
        for i in range(1, nx-1):
            matrix[0, j, i] = u0(x_0 + i * dx, y_0 + j * dy)

    # result_matrix = simulate_2d(nx, ny, nt, dx, dt, c_x, c_y, matrix)
    return simulate_2d_scheme2(nx, ny, nt, c_x, c_y, matrix)

@jit
def simulate_2d_scheme2(nx, ny, nt, c_x, c_y, u):
    for n in range(nt):  # loop for values of n from 0 to nt, so it will run nt times
        for i in range(1, nx):
            for j in range(1, ny):
                u[n + 1, i, j] = u[n, i, j] - c_x*(u[n, i, j] - u[n, i, j-1]) - c_y*(u[n, i, j] - u[n, i-1, j])
    return u
