# Courant-Friedrichs-Lewy stability #
def cfl_stability(constant, delta_x, delta_t):
    stability = constant * (delta_t / delta_x)

    if 0 < stability <= 1:
        print("Scheme: Stable")
        print(stability)
        return True
    else:
        print("Scheme: Unstable")
        print(str(stability) + ", needs to be over 0 and up to 1")
        return False


# Parabolic stability with explicit scheme.
# Euler method for time and centred second-order approximations for space
def pe_stability(constant, delta_x, delta_t):
    rs = (delta_x * delta_x) / (2 * constant)
    if delta_t <= rs:
        print("Scheme: Stable")
        return True
    else:
        print("Scheme: Unstable")
        print("Delta-t is too big compared to delta-x. ")
        return False

# def transport_equation(c, x, t, u0):
