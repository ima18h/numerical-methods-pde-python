# modules
import numpy as np
from matplotlib import pyplot as plt

# ---- Build [A] array and {B} column vector

m = 10  # size of array, make this 8000 to see time benefits

A = np.zeros((m, m), dtype=np.int)  # pre-allocate [A] array
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
print(A)
B[m - 1, 0] = 3

AA = np.zeros((m, m), dtype=np.int)
for i, v in enumerate((7,8,9)):
    np.fill_diagonal(AA[1:,i:], v)

AA[0,0] = 1
AA[0,1] = 2
AA[m - 1, m - 2] = 3
AA[m - 1, m - 1] = 4

print(AA)
# ---- Solve using numpy.linalg.solve

x = np.linalg.solve(A, B)  # solve A*x = B for x

plt.plot(np.linspace(0, 1, m), x)
