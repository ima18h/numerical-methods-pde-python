import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from equations import FOHyperbolicEquation
from equations import ParabolicEquation
from equations import SOWaveEquation
from equations import ReactionDiffusionEquation


#----------------------------Hyperbolic------------------------------------#
'''
# for FOHyperbolic, transport equation
# initial state at t0. This can be any similar function.
def u0(x):
    if 0 < x <= 1:
        return 1
    else:
        return 0


# Input arguments #
nx = 81  # Number of points in x direction.
nt = 41  # Number of points in t.

x_f = 2.0
dx = x_f / (nx - 1)  # Spacial step.

c = 1.0  # Constant.

dt = 0.025  # Time step.

# initialize, which then simulates if scheme is stable
u = FOHyperbolicEquation.initialize(nx, nt, dx, dt, c, u0)
#dt = 0.015
#uu = FOHyperbolicEquation.initialize(nx, nt, dx, dt, c, u0)

# analytical
nx_ana = 810
nt_ana = 500
dx_ana = x_f / (nx_ana - 1)
dt_ana = 0.0025
#u_ana = FOHyperbolicEquation.initialize_ana(nx_ana, nt_ana, dx_ana, dt_ana, c, u0)

# for plotting
#plt.axis('off')
#plt.plot(np.linspace(0, x_f, nx), u[0])
#plt.plot(np.linspace(0, x_f, nx), uu[0])
#plt.plot(np.linspace(0, x_f, nx_ana), u_ana[390])
#plt.legend(['dt = 0.025', 'dt = 0.015'])
plt.imshow(u, origin='lower')

#plt.plot(np.linspace(0, x_f, nx_ana), u_ana[200])
#plt.ylabel('u')
#plt.xlabel('x')
#plt.plot(np.linspace(0, x_f, nx), u[40])
'''
'''
for n in range(nt):
    plt.plot(np.linspace(0, x_f, nx), u[n])
    plt.pause(0.1)
'''
#------------------------------Parabolic----------------------------------#
'''
# Input arguments #
nx = 101  # Number of points in x direction.
nt = 201  # Number of points in t.

dt = 0.025  # Time step.
dx = 0.25  # Spacial step.

c = 1  # Constant.


# initial state at t0
def q(x, L):
    def H(x):
        if x >= 0:
            return 1
        else:
            return 0

    return H(x - ((5 * L) / 8)) - H(x - ((3 * L) / 8))


# initialize, which then simulates if scheme is stable
u = ParabolicEquation.initialize(nx, nt, dx, dt, c, q)
#plt.axis('off')
plt.plot(np.linspace(0, (nt-1)*dx, nx), u[0])
plt.plot(np.linspace(0, (nt-1)*dx, nx), u[51], linestyle='dashed')
plt.plot(np.linspace(0, (nt-1)*dx, nx), u[101], linestyle='dashed')
plt.plot(np.linspace(0, (nt-1)*dx, nx), u[151], linestyle='dashed')
plt.plot(np.linspace(0, (nt-1)*dx, nx), u[200])
#plt.imshow(u, origin='lower')
'''
'''
# for plotting
#plt.axis('off')
#for n in range(nt):
#    plt.plot(np.linspace(0, (nt-1)*dx, nx), u[n])
#    plt.pause(0.01)
'''
#-----------------------------Wave-----------------------------------#
'''
# Input arguments #
dx = 0.1
dt = 0.1

c = 1.0

t = 0.0
t_f = 10.0  # t final
x_0 = -5.0  # left boundary
x_f = 5.0  # right boundary
nx = int((x_f - x_0) / dx) + 1
nt = int((t_f - t) / dt) + 2


# initial state at t
def q(x):
    v = np.exp(-(x * x) / 0.25)
    return v


# initialize, which then simulates if scheme is stable
u = SOWaveEquation.initialize(nx, nt, dx, dt, c, q, x_0)
'''
'''

# for plotting
t = 7
tt = 10
n = int(t/dt)
nn = int(tt/dt)
#plt.axis('off')
plt.plot(np.linspace(-5, (nx - 1) * dx - 5, nx), u[n])
plt.plot(np.linspace(-5, (nx - 1) * dx - 5, nx), u[nn])
t = round(n * dt, 1)
t2 = round(nn * dt, 1)
#plt.legend(['t = ' + str(t), 't = ' + str(tt)])

'''
'''
plt.imshow(u, origin='lower')
'''
#--------------------------Parabolic--------------------------------------#
'''
# Input arguments #
nx = 101  # Number of points in x direction.
nt = 201  # Number of points in t.

dt = 0.025  # Time step.
dx = 0.25  # Spacial step.

c = 1  # Constant.


# initial state at t0
def q(x, L):
    def H(x):
        if x >= 0:
            return 1
        else:
            return 0

    return H(x - ((5 * L) / 8)) - H(x - ((3 * L) / 8))


# initialize, which then simulates if scheme is stable
u = ParabolicEquation.initialize(nx, nt, dx, dt, c, q)
u_ana = ParabolicEquation.initialize_ana(nx, nt, dx, dt, c, q)

# for plotting
#plt.axis('off')
#plt.plot(np.linspace(0, (nt-1)*dx, nx), u[1])
plt.plot(np.linspace(0, (nt-1)*dx, nx), u[0])
plt.plot(np.linspace(0, (nt-1)*dx, nx), u[51], linestyle='dashed')
plt.plot(np.linspace(0, (nt-1)*dx, nx), u[101], linestyle='dashed')
plt.plot(np.linspace(0, (nt-1)*dx, nx), u[151], linestyle='dashed')
plt.plot(np.linspace(0, (nt-1)*dx, nx), u[200])

plt.plot(np.linspace(0, (nt-1)*dx, nx), u_ana[0])
plt.plot(np.linspace(0, (nt-1)*dx, nx), u_ana[51], linestyle='dashed')
plt.plot(np.linspace(0, (nt-1)*dx, nx), u_ana[101], linestyle='dashed')
plt.plot(np.linspace(0, (nt-1)*dx, nx), u_ana[151], linestyle='dashed')
plt.plot(np.linspace(0, (nt-1)*dx, nx), u_ana[200])
#for n in range(1, nt):
#    if n % 67 == 0:
#        plt.plot(np.linspace(0, (nt-1)*dx, nx), u[n], linestyle='dashed')
'''
#-----------------------------Wave-----------------------------------#
'''
# Input arguments #
dx = 0.025
dy = 0.025
dt = 0.005

c = 1.0

t = 0.0
t_f = 10.0  # t final
x_0 = -5.0  # left boundary
x_f = 5.0  # right boundary
y_0 = -5.0  # left boundary
y_f = 5.0  # right boundary
nx = int((x_f - x_0) / dx) + 1
ny = int((y_f - y_0) / dy) + 1
nt = int((t_f - t) / dt) + 2

# initial state at t.
def q(x, y):
    v = np.exp((-(x * x) - (y * y)))*0.5
    return v

import time
start_time = time.clock()
# initialize, which then simulates if scheme is stable
u = SOWaveEquation.initialize_2d(nx, ny, nt, dx, dy, dt, c, q, x_0, y_0)
print("--- %s seconds ---" % (time.clock() - start_time))
#u = np.load('SOWaveEquation_2D_new2.npy')
#np.save('SOWaveEquation_2D_new2.npy', u)

# for plotting
t = 10.0
n = int(t/dt)

#plt.axis('off')
#plt.legend(['fucks sake dawg, know im sayn?'])
#plt.ylabel('u')
'''
'''
X = np.linspace(-5, 5, nx)
Y = np.linspace(-5, 5, ny)
X, Y = np.meshgrid(X, Y)
Z = u[n]
fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.set_axis_off()
ax.plot_wireframe(X, Y, Z, label='t = 5')
#ax.legend()

#surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=5)
'''
'''
#ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0)
#cset = ax.contour(X, Y, Z, zdir='u', offset=-0.15, cmap=cm.coolwarm)
#cset = ax.contour(X, Y, Z, zdir='x', offset=-5.5, cmap=cm.coolwarm)
#cset = ax.contour(X, Y, Z, zdir='y', offset=5.5, cmap=cm.coolwarm)

#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('u')
#ax.set_zlim(-0.25, 0.25)

plt.imshow(u[n], cmap=plt.cm.Reds, interpolation='none')
'''
#----------------------------Hyperbolic------------------------------------#
'''
# Input arguments #
dx = 0.1
dy = 0.1
dt = 0.01

c = 1.0

t = 0.0
t_f = 5.0  # t final
x_0 = -5.0  # left boundary
x_f = 5.0  # right boundary
y_0 = -5.0  # left boundary
y_f = 5.0  # right boundary
nx = int((x_f - x_0) / dx) + 1
ny = int((y_f - y_0) / dy) + 1
nt = int((t_f - t) / dt) + 1

# initial state at t.
def u0(x, y):
    v = np.exp((-(x * x) - (y * y)))*0.5
    return v


# initialize, which then simulates if scheme is stable
u = FOHyperbolicEquation.initialize_2d_scheme2(nx, ny, nt, dx, dy, dt, c, u0, x_0, y_0)
#u = np.load('FOHyperbolic_2D.npy')
#np.save('FOHyperbolic_2D_scheme2.npy', u)

# for plotting
t = 5
n = int(t/dt)

#plt.axis('off')
#plt.legend(['fucks sake dawg, know im sayn?'])
#plt.ylabel('u')

X = np.linspace(x_0, x_f, nx)
Y = np.linspace(y_0, y_f, ny)
X, Y = np.meshgrid(X, Y)
Z = u[n]
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_axis_off()
#ax.plot_wireframe(X, Y, Z)

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)


#ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0)
#cset = ax.contour(X, Y, Z, zdir='u', offset=-0.15, cmap=cm.coolwarm)
#cset = ax.contour(X, Y, Z, zdir='x', offset=-5.5, cmap=cm.coolwarm)
#cset = ax.contour(X, Y, Z, zdir='y', offset=5.5, cmap=cm.coolwarm)


#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('u')
#ax.set_zlim(-0.25, 0.25)


#plt.imshow(u[n], cmap=plt.cm.Reds, interpolation='none')

'''
#-----------------------------Parabolic-----------------------------------#
'''
# Input arguments #
dx = 0.1
dy = 0.1
dt = 0.01

c = 0.2

t = 0.0
t_f = 10.0  # t final
x_0 = -5.0  # left boundary
x_f = 5.0  # right boundary
y_0 = -5.0  # left boundary
y_f = 5.0  # right boundary
nx = int((x_f - x_0) / dx) + 1
ny = int((y_f - y_0) / dy) + 1
nt = int((t_f - t) / dt) + 1

# initial state at t.
def u0(x, y):
    v = np.exp((-(x * x) - (y * y)))*0.1
    return v


# initialize, which then simulates if scheme is stable
#u = ParabolicEquation.initialize(nx, ny, nt, dx, dy, dt, c, u0, x_0, y_0)
u = np.load('Parabolic_2D.npy')
#np.save('Parabolic_2D.npy', u)

# for plotting
t = 10
n = int(t/dt)

plt.axis('off')
#plt.legend(['fucks sake dawg, know im sayn?'])
#plt.ylabel('u')
'''
'''
X = np.linspace(x_0, x_f, nx)
Y = np.linspace(y_0, y_f, ny)
X, Y = np.meshgrid(X, Y)
Z = u[n]
fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.set_axis_off()
#ax.plot_wireframe(X, Y, Z)

#surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=5)


ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.4)
cset = ax.contour(X, Y, Z, zdir='u', offset=-0.01, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='x', offset=-5.5, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='y', offset=5.5, cmap=cm.coolwarm)


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('u')
#ax.set_zlim(-0.25, 0.25)

plt.imshow(u[n], cmap=plt.cm.afmhot, interpolation='none')

'''
'''
for i in range(int(nt/100)):
    plt.imshow(u[i*100], cmap=plt.cm.Reds, interpolation='none')
    plt.pause(0.5)
'''
#------------------------------pattern formation----------------------------------#
'''
# Input arguments #
dx = 1.0
dy = 1.0
dt = 1.0

D = 0.2
gamma = 0.5

t = 0.0
t_f = 200.0  # t final
x_0 = -250.0  # left boundary
x_f = 250.0  # right boundary
y_0 = -250.0  # left boundary
y_f = 250.0  # right boundary
nx = int((x_f - x_0) / dx) + 1
ny = int((y_f - y_0) / dy) + 1
nt = int((t_f - t) / dt) + 1

# Not used here
def u0(x, y):
    return


# initialize, which then simulates if scheme is stable
#u = ReactionDiffusionEquation.initialize_2d(nx, ny, nt, dx, dy, dt, D, gamma, u0, x_0, y_0, randInit=True)
u = np.load('PatternFormation.npy')
#np.save('PatternFormation.npy', u)

# for plotting
t = 200
n = int(t/dt)

#plt.axis('off')
#plt.legend(['t = 150'])
#plt.ylabel('u')
'''
'''
'''
'''

X = np.linspace(x_0, x_f, nx)
Y = np.linspace(y_0, y_f, ny)
X, Y = np.meshgrid(X, Y)
Z = u[n]
fig = plt.figure()
ax = fig.gca(projection='3d')

#ax.set_axis_off()
#ax.plot_wireframe(X, Y, Z)

#surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=5)


#ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.4)
#cset = ax.contour(X, Y, Z, zdir='u', offset=-0.01, cmap=cm.coolwarm)
#cset = ax.contour(X, Y, Z, zdir='x', offset=-5.5, cmap=cm.coolwarm)
#cset = ax.contour(X, Y, Z, zdir='y', offset=5.5, cmap=cm.coolwarm)


#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('u')
#ax.set_zlim(-0.25, 0.25)
'''

'''
plt.axis('off')
plt.imshow(u[n], cmap=plt.cm.binary, interpolation='none', origin='lower')


'''
#for i in range(int(nt/10)):
#    plt.imshow(u[i*10], cmap=plt.cm.binary, interpolation='none', origin='lower')
#    plt.pause(1)
'''
'''
#-----------------------------Flow / Diffusion 1d-----------------------------------#
'''
# Input arguments #
dx = 0.1
dt = 0.1

D = 0.2

t = 0.0
t_f = 20.0  # t final
x_0 = -25.0  # left boundary
x_f = 25.0  # right boundary
nx = int((x_f - x_0) / dx) + 1
nt = int((t_f - t) / dt) + 1

# Not used here
def u0(x):
    return -0.01*x


# initialize, which then simulates if scheme is stable
#u = ReactionDiffusionEquation.initialize(nx, nt, dx, dt, D, u0, x_0, randInit=False)
u = np.load('diffusion1d.npy')
#np.save('diffusion1d.npy', u)

# for plotting
t = 10
n = int(t/dt)

#plt.axis('off')
#plt.legend(['t = 150'])
#plt.ylabel('u')

#X = np.linspace(x_0, x_f, nx)
#Y = np.linspace(y_0, y_f, ny)
#X, Y = np.meshgrid(X, Y)
#Z = u[n]
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.set_axis_off()
#ax.plot_wireframe(X, Y, Z)

#surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=5)


#ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.4)
#cset = ax.contour(X, Y, Z, zdir='u', offset=-0.01, cmap=cm.coolwarm)
#cset = ax.contour(X, Y, Z, zdir='x', offset=-5.5, cmap=cm.coolwarm)
#cset = ax.contour(X, Y, Z, zdir='y', offset=5.5, cmap=cm.coolwarm)


#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('u')
#ax.set_zlim(-0.25, 0.25)

plt.plot(np.linspace(x_0, x_f, nx), u[n])
'''

'''
ratio = 10
nn = int(nt / ratio)
for n in range(nn):
    plt.plot(np.linspace(x_0, x_f, nx), u[n * ratio])
    plt.pause(1.5)
'''
#-----------------------------pattern formation system-----------------------------------#
'''
# Input arguments #
dx = 1.0
dy = 1.0
dt = 1.0

Du = 0.2
Dv = 0.1
alpha = 1.0
beta = 0.5
gamma = 0.5

t = 0.0
t_f = 200.0  # t final
x_0 = 0.0  # left boundary
x_f = 200.0  # right boundary
y_0 = 0.0  # left boundary
y_f = 200.0  # right boundary
nx = int((x_f - x_0) / dx) + 1
ny = int((y_f - y_0) / dy) + 1
nt = int((t_f - t) / dt) + 1

# Not used here
def u0(x, y):
    return


# initialize, which then simulates if scheme is stable
u = ReactionDiffusionEquation.initialize_sys(nx, ny, nt, dx, dy, dt, Du, Dv, alpha, gamma, beta, u0, x_0, y_0, randInit=True)
#u = np.load('PatternFormation.npy')
#np.save('PatternFormation.npy', u)

# for plotting
t = 100
n = int(t/dt)

plt.axis('off')
#plt.legend(['t = 150'])
#plt.ylabel('u')
'''
'''
X = np.linspace(x_0, x_f, nx)
Y = np.linspace(y_0, y_f, ny)
X, Y = np.meshgrid(X, Y)
Z = u[n]
fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.set_axis_off()
#ax.plot_wireframe(X, Y, Z)

#surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=5)


ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.4)
cset = ax.contour(X, Y, Z, zdir='u', offset=-0.01, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='x', offset=-5.5, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='y', offset=5.5, cmap=cm.coolwarm)


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('u')
#ax.set_zlim(-0.25, 0.25)
'''
#plt.imshow(u[n], cmap=plt.cm.binary, interpolation='none', origin='lower')


'''
for i in range(int(nt/10)):
    plt.imshow(u[i*10], cmap=plt.cm.binary, interpolation='none', origin='lower')
    plt.pause(1)
'''
'''
a = 2.8e-4
b = 5e-3
tau = .1
k = -.005

size = 10  # size of the 2D grid
dx = 2. / size  # space step
T = 9.0  # total time
dt = .001  # time step
n = int(T / dt)  # number of iterations

U = np.random.rand(size, size)
U = np.zeros((size, size))
U[4, -4] = 1
V = np.random.rand(size, size)
kk = U

def laplacian(Z):
    Ztop = Z[0:-2, 1:-1]
    Zleft = Z[1:-1, 0:-2]
    Zbottom = Z[2:, 1:-1]
    Zright = Z[1:-1, 2:]
    Zcenter = Z[1:-1, 1:-1]
    return (Ztop + Zleft + Zbottom + Zright -
            4 * Zcenter) / dx**2

deltaU = laplacian(U)
kk = kk[1:-1, 1:-1]
u = kk


for j in range(1, size-3):  # loop for values of n from 0 to nt, so it will run nt times
    for i in range(1, size-3):
        kk[j, i] = (u[j, i+1] + u[j, i-1] + u[j+1, i] + u[j-1, i] - (4*u[j, i])) / dx**2

print(deltaU)
print(kk)
'''
'''
size = 201  # size of the 2D grid
dx = 1  # space step
T = 200.0  # total time
dt = 1  # time step
n = int(T / dt)  # number of iterations

t = 200
n = int(t/dt)

u = np.load('ReactionDiffusionSystem.npy')


plt.axis('off')
plt.imshow(u[n], cmap=plt.cm.copper, interpolation='bilinear', origin='lower')
'''
