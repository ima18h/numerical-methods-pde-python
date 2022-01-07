import numpy as np
from util import cfl_stability as stability
from numba import jit


def initialize(nx, nt, dx, dt, c, u0, x_0):
    stability(c, dx, dt)
    matrix = np.zeros([nt, nx])

    for j in range(1, nx - 1):
        matrix[0, j] = u0(x_0 + j * dx)
        #matrix[1, j] = matrix[0, j]

    # for first time-step
    for j in range(1, nx-1):
        matrix[1, j] = matrix[0, j] - 0.5*c*c*(matrix[0, j + 1] - (2 * matrix[0, j]) + matrix[0, j - 1])

    result_matrix = simulate(nx, nt, dx, dt, c, matrix)
    return result_matrix


# Explicit scheme.
#
def simulate(nx, nt, dx, dt, c, u):
    constant = (c * c) * (dt * dt) / (dx * dx)
    for n in range(1, nt - 1):  # loop for values of n from 0 to nt, so it will run nt times
        for j in range(1, nx - 1):
            u[n + 1, j] = constant * (u[n, j + 1] - (2 * u[n, j]) + u[n, j - 1]) + (2 * u[n, j]) - u[n - 1, j]
    return u
#shit
def initialize_ana(nx, nt, dx, dt, c, u0, x_0):
    matrix = np.zeros([nt, nx])

    for j in range(1, nx - 1):
        matrix[0, j] = u0(x_0 + j * dx)
        #matrix[1, j] = matrix[0, j]

    # for first time-step
    for j in range(1, nx-1):
        matrix[1, j] = 5

    return simulate_ana(nx, nt, dx, dt, c, matrix)
#shit
def simulate_ana(nx, nt, dx, dt, c, u):
    constant = (c * c) * (dt * dt) / (dx * dx)
    for n in range(1, nt - 1):  # loop for values of n from 0 to nt, so it will run nt times
        for j in range(1, nx - 1):
            u[n + 1, j] = 5
    return u


def initialize_2d(nx, ny, nt, dx, dy, dt, c, u0, x_0, y_0):
    matrix = np.zeros((nt, ny, nx))
    c_x = (c * c * dt * dt) / (dx * dx)
    c_y = (c * c * dt * dt) / (dy * dy)

    for i in range(1, ny-1):
        for j in range(1, nx-1):
            matrix[0, i, j] = u0(x_0 + j * dx, y_0 + i * dy)
            matrix[1, i, j] = matrix[0, i, j] + dt*2*matrix[0, i, j] + 0.5*c_x*(matrix[0, i, j+1] - 2*matrix[0, i, j] + matrix[0, i, j-1]) + 0.5*c_y*(matrix[0, i+1, j] - 2*matrix[0, i, j] + matrix[0, i-1, j])

    # result_matrix = simulate_2d(nx, ny, nt, dx, dt, c_x, c_y, matrix)
    return simulate_2d(nx, ny, nt, c_x, c_y, matrix)

@jit
def simulate_2d(nx, ny, nt, c_x, c_y, u):
    for n in range(1, nt - 1):  # loop for values of n from 0 to nt, so it will run nt times
        for i in range(1, nx - 2):
            for j in range(1, ny - 2):
                u[n + 1, i, j] = (c_x * (u[n, i, j+1] - (2 * u[n, i, j]) + u[n, i, j-1])) + (c_y * (u[n, i+1, j] - (2 * u[n, i, j]) + u[n, i-1, j])) + (2 * u[n, i, j]) - u[n - 1, i, j]
    return u
