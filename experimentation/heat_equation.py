import numpy as np
from matplotlib import pyplot
from math import sin
from math import pi

# Domain #
x_f = 1
t_f = 10

# Grid #
DeltaX = 1 / 6
DeltaT = 1 / 80

    # Number of points #
j = int(x_f / DeltaX) - 1
m = int(t_f / DeltaT)

# Setup #
def f(x):
    return sin(2 * pi * x)



#  #
np.ones(j)

# Simulation #
