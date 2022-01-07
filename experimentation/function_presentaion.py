from numpy import e, linspace, pi, vectorize, array, dot, diag, zeros, ones, concatenate
from math import sin
from matplotlib import animation, pyplot as plt
from time import time


# Here are the important functions
def EulerHeatFunctionBC(u0, x, fl, fr, dt, nr_times, t0):
    """ This function solves the heat equation :
                    u_t = u_xx
    by using the Euler method. The function takes
    an initial condition u0, a domain  x, boundar conditions fl & fr
    a time step dt, the number of times to run the loop,
    and the initial time, t0, as input.
    """
    # Important Constants, etc.
    time = t0
    dx = x[1] - x[0]
    numUnknowns = len(u0) - 2
    r = dt / (dx ** 2)
    mainDiagonal = -2 * ones(numUnknowns)
    offDiagonal = ones(numUnknowns - 1)  # Note that we need one less element.
    T = MakeTridiagonalMatrix(mainDiagonal,
                              offDiagonal)  # We could consider a better place to make T to make the code faster.
    u = u0

    # Make the boundary condition vector
    BoundaryConditions = zeros(numUnknowns)
    BoundaryConditions[0] = u[0]
    BoundaryConditions[-1] = u[-1]

    # Loop to perform the calculations
    for step in range(nr_times):
        time = time + dt  # We could reconsider how to do this to minimze rounding errors.
        u[1:-1] = u[1:-1] + r * dot(T, u[1:-1]) + r * BoundaryConditions
        BoundaryConditions[0] = fl(time)
        BoundaryConditions[-1] = fr(time)
        u[0] = fl(time)
        u[-1] = fr(time)

    return u, time


def MakeTridiagonalMatrix(main, offset_one):
    """This function will make a tridiagonal 2D array (NOT a matrix)
    which has the main array on its main diagonal and the offset_one
    array on its super and sub diagonals.
    """
    return diag(main) + diag(offset_one, k=-1) + diag(offset_one, k=1)


# Here we set some variables
x = linspace(-pi / 2, pi / 2, 10)
f = lambda x, t: e ** (-t) * sin(x) + 2  # An "anonymous function".
fl = lambda t: f(x[0], t)
fr = lambda t: f(x[-1], t)
F = vectorize(f)  # Makes a function work on a vector of values.
t0 = 0
dx = x[1] - x[0]
r = .49  # This can be as high as 0.5.
dt = r * (dx ** 2)
u0 = F(x, t0)
u = u0
time = t0
finaltime = 6.0

# Set up the figure
fig = plt.figure()
ax = plt.axes(xlim=(-pi / 2, pi / 2), ylim=(1, 3))
ax.set_title('Euler Heat Equation Solver Demo', color='black')
ax.set_xlabel('X')
ax.set_ylabel('u(x,t) = Temperature')
line, = ax.plot([], [], lw=2)  # Initialize the line and give it a thickness
time_label = ax.text(0.05, 0.90, '', transform=ax.transAxes, fontsize=14)  # initialize the time label for the graph


# The use of transform=ax.transAxes throughout the code indicates that the
# coordinates are given relative to the axes bounding box, with 0,0 being
# the lower left of the axes and 1,1 the upper right.

# Initialization function
def init():
    time_label.set_text('')
    line.set_data([], [])
    return line, time_label


# Animation function.  
def update(i):
    global u, time  # Necessary.
    nr_times = 1
    u, time = EulerHeatFunctionBC(u, x, fl, fr, dt, nr_times, time)
    time_label.set_text('time = %.3f' % time)  # Display the current time to the accuracy of your liking.
    line.set_data(x, u)  # Set the data in the line
    return line, time_label


# Start the animation
HeatAnimation = animation.FuncAnimation(fig, update, frames=int(finaltime / dt), interval=100, init_func=init,
                                        blit=False, repeat=False)

plt.show()

print(max(abs(u - F(x, time))))  # Print out the error in the numerical method.
