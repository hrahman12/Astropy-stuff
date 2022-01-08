import numpy as np
from scipy.constants import G

class Body:

    def __init__(self, m, name=None):
        self.m = m
        self.name = name

        # protected attributes
        self._x = np.zeros(3)
        self._v = np.zeros(3)

    def print_mass(self):
        if self.name == None:
            print(f"Mass = {self.m:.2e} kg")
        else:
            print("Mass of", self.name, f"= {self.m:.2e} kg")

    def set_state(self, x0, v0):
        # ensure x0 and v0 are arrays
        x0 = np.array(x0); v0 = np.array(v0)

        # accept only if there are three elements
        try:
            if x0.size == 3 and v0.size == 3:
                self._x = x0
                self._v = v0
            else:
                raise ValueError
        except ValueError:
            print("Invalid argument: must be array-like with three elements")

    def pos(self):
        return self._x

    def vel(self):
        return self._v

    # compute distance between this body and another
    def distance(self, body):
        try:
            if isinstance(body, Body):
                return ((self._x[0] - body._x[0])**2 +
                        (self._x[1] - body._x[1])**2 +
                        (self._x[2] - body._x[2])**2)**(1/2)
            else:
                raise TypeError
        except TypeError:
            print("Invalid argument: must be instance of Body")

    # compute gravitational force exerted by another body
    def gravity(self, body):
        delta =  body._x - self._x # distance vector
        return G * self.m * body.m * delta / np.sum(delta*delta)**(3/2)

    @classmethod
    def two_body_step(cls, body1, body2, dt):
        """
        symplectic Euler step for the two-body problem

        args: body1, body2 - the two bodies
              dt - time step
        """
        force = cls.gravity(body1, body2)

        body1._v += force * dt / body1.m
        body2._v -= force * dt / body2.m

        body1._x += body1._v * dt
        body2._x += body2._v * dt
