# modules
import numpy as np
import time

m = 10
# Create arrays and set values
ab = np.zeros((3, m))
b = 2 * np.ones(m)
ab[0] = 9
ab[1] = 8
ab[2] = 7

# Fix end points
ab[0, 1] = 2
ab[1, 0] = 1
ab[1, -1] = 4
ab[2, -2] = 3
b[0] = 1
b[-1] = 3

print(ab)
print(b)

A = np.zeros((m, m))  # pre-allocate [A] array
B = np.zeros((m, 1))  # pre-allocate {B} column vector

A[0, 0] = 1
A[0, 1] = 2
B[0, 0] = 1

for i in range(1, m - 1):
    A[i, i - 1] = 7  # node-1
    A[i, i] = 8  # node
    A[i, i + 1] = 9  # node+1
    B[i, 0] = 2

A[m - 1, m - 2] = 3
A[m - 1, m - 1] = 4
B[m - 1, 0] = 3

print(A)
print(B)
