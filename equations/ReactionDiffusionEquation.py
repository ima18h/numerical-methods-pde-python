import numpy as np
import random
from numba import jit


def initialize(nx, nt, dx, dt, D, f0, x_0, randInit=True):
    matrix = np.zeros((nt, nx))
    c_x = (D * dt) / (dx * dx)

    if randInit:
        for i in range(nx):
            matrix[0, i] = random.random()
    else:
        for i in range(nx):
            matrix[0, i] = f0(x_0 + i * dx)

    # result_matrix = simulate_2d(nx, ny, nt, dx, dt, c_x, c_y, matrix)
    return simulate(nx, nt, c_x, matrix)


@jit
def simulate(nx, nt, c_x, u):
    for n in range(nt - 1):  # loop for values of n from 0 to nt, so it will run nt times
        for i in range(1, nx - 1):
            u[n + 1, i] = u[n, i] + (c_x * (u[n, i + 1] - (2 * u[n, i]) + u[n, i - 1]))
    return u


def initialize_2d(nx, ny, nt, dx, dy, dt, D, y, f0, x_0, y_0, randInit=True):
    matrix = np.zeros((nt, ny, nx))
    c_x = (D * dt) / (dx * dx)
    c_y = (D * dt) / (dy * dy)

    if randInit:
        for j in range(ny):
            for i in range(nx):
                matrix[0, j, i] = random.random()
    else:
        for j in range(1, ny):
            for i in range(nx):
                matrix[0, j, i] = f0(x_0 + i * dx, y_0 + j * dy)

    matrix[0, -1, :] = matrix[0, 0, :]
    matrix[0, :, -1] = matrix[0, :, 0]
    # result_matrix = simulate_2d(nx, ny, nt, dx, dt, c_x, c_y, matrix)
    return simulate_2d(nx, ny, nt, c_x, c_y, y, matrix)


@jit
def simulate_2d(nx, ny, nt, c_x, c_y, y, u):
    # Central difference space and explicit Euler for time
    for n in range(nt - 1):  # loop for values of n from 0 to nt, so it will run nt times
        # loop for middle of matrix
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                u[n + 1, j, i] = u[n, j, i] + (c_x * (u[n, j, i + 1] - (2 * u[n, j, i]) + u[n, j, i - 1])) + (
                        c_y * (u[n, j + 1, i] - (2 * u[n, j, i]) + u[n, j - 1, i])) + y * u[n, j, i] - y * u[n, j, i] * \
                                 u[n, j, i]

        # rest are for boundaries...
        for j in range(1, ny - 1):
            u[n + 1, j, 0] = u[n, j, 0] + (c_x * (u[n, j, 1] - (2 * u[n, j, 0]) + u[n, j, -2])) + (
                    c_y * (u[n, j + 1, 0] - (2 * u[n, j, 0]) + u[n, j - 1, 0])) + y * u[n, j, 0] - y * u[n, j, 0] * u[
                                 n, j, 0]
            u[n + 1, j, -1] = u[n + 1, j, 0]

        for i in range(1, nx - 1):
            u[n + 1, 0, i] = u[n, 0, i] + (c_x * (u[n, 0, i + 1] - (2 * u[n, 0, i]) + u[n, 0, i - 1])) + (
                    c_y * (u[n, 1, i] - (2 * u[n, 0, i]) + u[n, -2, i])) + y * u[n, 0, i] - y * u[n, 0, i] * u[n, 0, i]
            u[n + 1, -1, i] = u[n + 1, 0, i]

        u[n + 1, 0, 0] = u[n, 0, 0] + (c_x * (u[n, 0, 1] - (2 * u[n, 0, 0]) + u[n, 0, -2])) + (
                c_y * (u[n, 1, 0] - (2 * u[n, 0, 0]) + u[n, -2, 0])) + y * u[n, 0, 0] - y * u[n, 0, 0] * u[
                             n, 0, 0]
        u[n + 1, -1, 0] = u[n, -1, 0] + (c_x * (u[n, -1, 1] - (2 * u[n, -1, 0]) + u[n, -1, -2])) + (
                c_y * (u[n, 1, 0] - (2 * u[n, -1, 0]) + u[n, -2, 0])) + y * u[n, -1, 0] - y * u[n, -1, 0] * u[n, -1, 0]
        u[n + 1, -1, -1] = u[n, -1, -1] + (c_x * (u[n, -1, 1] - (2 * u[n, -1, -1]) + u[n, -1, -2])) + (
                c_y * (u[n, 1, -1] - (2 * u[n, -1, -1]) + u[n, -2, -1])) + y * u[n, -1, -1] - y * u[n, -1, -1] * u[
                               n, -1, -1]
        u[n + 1, 0, -1] = u[n + 1, 0, -1] + (c_x * (u[n, 0, 1] - (2 * u[n + 1, 0, -1]) + u[n, 0, -2])) + (
                c_y * (u[n, 1, -1] - (2 * u[n + 1, 0, -1]) + u[n, -2, -1])) + y * u[n + 1, 0, -1] - y * u[
                              n + 1, 0, -1] * u[n + 1, 0, -1]
    return u


def initialize_sys(nx, ny, nt, dx, dy, dt, Du, Dv, a, y, b, f0, x_0, y_0, randInit=True):
    u = np.zeros((nt, ny, nx))
    v = np.zeros((nt, ny, nx))

    cu_x = (Du * dt) / (dx * dx)
    cu_y = (Du * dt) / (dy * dy)
    cv_x = (Dv * dt) / (dx * dx)
    cv_y = (Dv * dt) / (dy * dy)

    if randInit:
        for j in range(ny):
            for i in range(nx):
                u[0, j, i] = random.random()
                v[0, j, i] = random.random()
    else:
        for j in range(1, ny - 1):
            for i in range(nx):
                u[0, j, i] = f0(x_0 + i * dx, y_0 + j * dy)
                v[0, j, i] = f0(x_0 + i * dx, y_0 + j * dy)

    # result_matrix = simulate_2d(nx, ny, nt, dx, dt, cu_x, cu_y, u)
    return simulate_sys(nx, ny, nt, cu_x, cu_y, cv_x, cv_y, y, u, v)


@jit
def simulate_sys(nx, ny, nt, cu_x, cu_y, cv_x, cv_y, y, u, v):
    for n in range(nt - 1):  # loop for values of n from 0 to nt, so it will run nt times
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                u[n + 1, j, i] = u[n, j, i] + (cu_x * (u[n, j, i + 1] - (2 * u[n, j, i]) + u[n, j, i - 1])) + (
                        cu_y * (u[n, j + 1, i] - (2 * u[n, j, i]) + u[n, j - 1, i])) + y * u[n, j, i] - y * u[n, j, i] * \
                                 u[n, j, i]
    return u
