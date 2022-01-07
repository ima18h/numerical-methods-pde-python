import numpy as np
from util import pe_stability as stability
from numba import jit
from math import sqrt, exp


def initialize(nx, nt, dx, dt, c, u0):
    stable = stability(c, dx, dt)
    if stable:
        matrix = np.zeros([nt, nx])

        x_f = (nx - 1) * dx
        for j in range(1, nx - 1):
            matrix[0, j] = -u0(j * dx, x_f)  # this is giving me -1 instead of 1 so putting - in front for now

        result_matrix = simulate(nx, nt, dx, dt, c, matrix)
        return result_matrix
    else:
        exit()


# Explicit scheme.
# Euler method for time and centred second-order approximations for space
def simulate(nx, nt, dx, dt, c, u):
    for n in range(1, nt):  # loop for values of n from 0 to nt, so it will run nt times
        for j in range(1, nx - 1):
            u[n, j] = u[n - 1, j] + (c * (dt / (dx * dx))) * (u[n - 1, j - 1] - (2 * u[n - 1, j]) + u[n - 1, j + 1])
    return u

#wrong
def initialize_ana(nx, nt, dx, dt, c, u0):
    stable = stability(c, dx, dt)
    if stable:
        matrix = np.zeros([nt, nx])

        x_f = (nx - 1) * dx
        for j in range(1, nx - 1):
            matrix[0, j] = -u0(j * dx, x_f)  # this is giving me -1 instead of 1 so putting - in front for now

        return simulate_ana(nx, nt, dx, dt, c, matrix)
    else:
        exit()

# wrong
def simulate_ana(nx, nt, dx, dt, c, u):
    for n in range(1, nt):  # loop for values of n from 0 to nt, so it will run nt times
        for j in range(1, nx - 1):
            u[n, j] = (1 / sqrt(1 + 4 * (n*dt))) * exp((-(dx*j)*dx*j) / (1 + 4*(n*dt)))
    return u


# implicit scheme
def initialize_imp(nx, nt, dx, dt, c, u0):
    stability(c, dx, dt)
    # To solving, Ax = b
    A = np.zeros((nx, nx))
    x = np.zeros((nx, 1))
    b = np.zeros((nx, 1))

    gamma = dt / (dx * dx)
    diagConstant = (2 * gamma) + 1

    for i, v in enumerate((-gamma, diagConstant, -gamma)):
        np.fill_diagonal(A[1:, i:], v)

    A[0, 0] = diagConstant
    A[0, 1] = -gamma

    x_f = (nx - 1) * dx
    for j in range(1, nx - 1):
        b[j, 0] = -u0(j * dx, x_f)

    # --------------------------------------------------
    u = np.zeros((nt, nx))
    u[0] = np.transpose(b)
    # --------------------------------------------------
    result_matrix = simulate_imp(nt, nx, A, x, u)
    return result_matrix


# backwards Euler method, implicit
def simulate_imp(nt, nx, A, x, u):
    for n in range(0, nt - 1):
        x = np.linalg.solve(A, u[n])
        u[n + 1] = np.transpose(x)
        u[n + 1, 0] = 0
        u[n + 1, nx - 1] = 0
    return u


def initialize_2d(nx, ny, nt, dx, dy, dt, c, u0, x_0, y_0):
    matrix = np.zeros((nt, ny, nx))
    c_x = (c * dt) / (dx * dx)
    c_y = (c * dt) / (dy * dy)

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            matrix[0, j, i] = u0(x_0 + i * dx, y_0 + j * dy)

    # result_matrix = simulate_2d(nx, ny, nt, dx, dt, c_x, c_y, matrix)
    return simulate_2d(nx, ny, nt, c_x, c_y, matrix)


@jit
def simulate_2d(nx, ny, nt, c_x, c_y, u):
    for n in range(nt - 1):  # loop for values of n from 0 to nt, so it will run nt times
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                u[n + 1, j, i] = u[n, j, i] + (c_x * (u[n, j, i + 1] - (2 * u[n, j, i]) + u[n, j, i - 1])) + (
                        c_y * (u[n, j + 1, i] - (2 * u[n, j, i]) + u[n, j - 1, i]))
    return u
