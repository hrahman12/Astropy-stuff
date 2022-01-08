cpdef double crk4_step(f, double t, double x, double dt):

    cdef double k1 = dt * f(t, x)
    cdef double k2 = dt * f(t + 0.5*dt, x + 0.5*k1)
    cdef double k3 = dt * f(t + 0.5*dt, x + 0.5*k2)
    cdef double k4 = dt * f(t + dt, x + k3)

    return x + (k1 + 2*(k2 + k3) + k4)/6.0

cdef double rdot(double t, double r):
    return (1.0 - r**3)/(3.0*r**2)

cpdef double stroemgren_step(double t, double x, double dt):

    cdef double k1 = dt * rdot(t, x)
    cdef double k2 = dt * rdot(t + 0.5*dt, x + 0.5*k1)
    cdef double k3 = dt * rdot(t + 0.5*dt, x + 0.5*k2)
    cdef double k4 = dt * rdot(t + dt, x + k3)

    return x + (k1 + 2*(k2 + k3) + k4)/6.0
