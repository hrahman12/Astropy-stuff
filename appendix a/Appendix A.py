#!/usr/bin/env python
# coding: utf-8

# # Appendix A: Object-Oriented Programming

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '1')
get_ipython().run_line_magic('aimport', 'nbody')

from astropy.constants import M_sun

body1 = nbody.Body(2.06*M_sun.value, "Sirus A")


# In[ ]:


print("Mass of", body1.name, f"= {body1.m:.2e} kg")


# In[ ]:


body2 = nbody.Body(1.02*M_sun.value, "Sirus B")

body2.print_mass()


# In[ ]:


from math import pi
from scipy.constants import au,G

M1 = body1.m
M2 = body2.m

# orbital parameters
a = 2.64*7.4957*au
e = 0.5914         
T = pi * (G*(M1 + M2))**(-1/2) * a**(3/2)

# periastron
d = a*(1 - e)
v = (G*(M1 + M2)*(2/d - 1/a))**(1/2) # vis-viva eq.

body1.set_state([d*M2/(M1 + M2), 0], [0, -v*M2/(M1 + M2)])


# In[ ]:


body1.set_state([d*M2/(M1 + M2), 0, 0], 
                [0, -v*M2/(M1 + M2), 0])
body2.set_state([-d*M1/(M1 + M2), 0, 0], 
                [0, v*M1/(M1 + M2), 0])


# In[ ]:


print(body1.pos())


# In[ ]:


print("{:.2f} AU, {:.2f} AU".
      format(d/au, body1.distance(body2)/au))


# In[ ]:


print(nbody.Body.distance(body1, body2)/au)


# In[ ]:


import numpy as np

n_rev = 3      # number of revolutions
n = n_rev*500  # number of time steps
dt = n_rev*T/n # time step
t = np.arange(0, (n+1)*dt, dt)


# In[ ]:


orbit1 = np.zeros([n+1,3])
orbit2 = np.zeros([n+1,3])

# integrate two-body problem
for i in range(n+1):
    orbit1[i] = body1.pos()
    orbit2[i] = body2.pos()
    
    nbody.Body.two_body_step(body1, body2, dt)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

fig = plt.figure(figsize=(6, 6*25/35), dpi=100)

plt.plot([0], [0], '+k')  # center of mass
plt.plot(orbit1[:,0]/au, orbit1[:,1]/au, color='red', label='Sirius A')
plt.plot(orbit2[:,0]/au, orbit2[:,1]/au, color='blue', label='Sirius B')

plt.xlabel("$x$ [AU]")
plt.xlim(-12.5,22.5)
plt.ylabel("$y$ [AU]")
plt.ylim(-12.5,12.5)
plt.legend(loc='upper left')


# In[ ]:




