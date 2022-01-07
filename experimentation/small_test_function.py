dx = 0.3
xf = 60 * dx
x = -20 * dx*0


def H(x):
    if x >= 0:
        return 1
    else:
        return 0


print(H(x - ((5 * xf) / 8)) - H(x - ((3 * xf) / 8)))
print(H(x - ((5 * xf) / 8)) - H(x - ((3 * xf) / 8)))
