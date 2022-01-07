import numpy as np
import matplotlib.pyplot as plt

Du = 0.2
Dv = 0.1
a = 1
b = 0.5
y = 0.5

size = 200  # size of the 2D grid
dx = 1  # space step
T = 200.0  # total time
dt = 1  # time step
n = int(T / dt)  # number of iterations

U = np.random.rand(size, size)
U[0, :] = U[-1, :]
U[:, 0] = U[:, -1]
V = np.random.rand(size, size)
V[0, :] = U[-1, :]
V[:, 0] = U[:, -1]
u = np.zeros((n + 2, size, size))
u[0] = U


def laplacian(Z):
    top = Z[0:-2, 1:-1]
    left = Z[1:-1, 0:-2]
    bottom = Z[2:, 1:-1]
    right = Z[1:-1, 2:]
    center = Z[1:-1, 1:-1]
    return (top + left + bottom + right -
            4 * center) / (dx * dx)


def show_patterns(U, ax=None):
    ax.imshow(U, cmap=plt.cm.copper,
              interpolation='bilinear',
              extent=[-1, 1, -1, 1])
    ax.set_axis_off()


fig, axes = plt.subplots(3, 3, figsize=(8, 8))
step_plot = n // 9
# We simulate the PDE with the finite difference
# method.
for i in range(n):
    # We compute the Laplacian of u and v.
    deltaU = laplacian(U)
    deltaV = laplacian(V)
    # We take the values of u and v inside the grid.
    Uc = U[1:-1, 1:-1]
    Vc = V[1:-1, 1:-1]
    # We update the variables.
    U[1:-1, 1:-1], V[1:-1, 1:-1] = \
        Uc + dt * (Du * deltaU + y * Uc - y * Uc ** 2), \
        Vc + dt * (Dv * deltaV + b * Uc - b * a * Vc * Uc ** 2)

    # Neumann conditions: derivatives at the edges
    # are null.
    for Z in (U, V):
        Z[0, :] = Z[1, :]
        Z[-1, :] = Z[-2, :]
        Z[:, 0] = Z[:, 1]
        Z[:, -1] = Z[:, -2]

    u[i + 1] = U

    # We plot the state of the system at
    # 9 different times.
    if i % step_plot == 0 and i < 9 * step_plot:
        ax = axes.flat[i // step_plot]
        show_patterns(U, ax=ax)
        ax.set_title(f'$t={i * dt:.2f}$')

np.save('ReactionDiffusionSystem2.npy', u)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
show_patterns(U[1:-1, 1:-1], ax=ax)
