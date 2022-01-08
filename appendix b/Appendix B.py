#!/usr/bin/env python
# coding: utf-8

# # Appendix B: Making Python Faster

# ## Using arrays

# #### Data types

# In[ ]:


N = [79, 171, 265, 355]


# In[ ]:


N[1], type(N[1]), type(N)


# In[ ]:


import numpy as np

# convert to array
N = np.array(N)

print(N[1], N[1].dtype, N.dtype)


# In[ ]:


# redefine list 
N = [79, "summer solstice", 265, "winter solstice"]

N[1], type(N[1]), type(N)


# In[ ]:


N[0], type(N[0]), type(N)


# #### Example: Planck spectrum

# Compute a Planck spectrum using lists

# In[ ]:


import math
from scipy.constants import h,c,k,sigma


# In[ ]:


# list of wavenumbers
n = 1000
lambda_max = 2e-6
lambda_step = lambda_max/n
wavelength = [i*lambda_step for i in range(1,n+1)]


# In[ ]:


def planck_spectrum(wavelength, T=5778):
    
    # create empty list
    spectrum = []

    # loop over wavelengths and append flux values
    for val in wavelength:
        spectrum.append(2*h*c**2 / 
            (val**5 * (math.exp(min(700, h*c/(val*k*T))) - 1)))
        
    return spectrum


# In[ ]:


get_ipython().run_line_magic('timeit', 'planck_spectrum(wavelength)')


# Compute a Planck spectrum using arrays

# In[ ]:


import numpy as np

# array of wavenumbers
n = 1000
lambda_max = 2e-6
wavelength = np.linspace(lambda_max/n, lambda_max, n)


# In[ ]:


def planck_spectrum(wavelength, T=5778):
    return 2*h*c**2 / (wavelength**5 * 
        (np.exp(np.minimum(700, h*c/(wavelength*k*T))) - 1))


# In[ ]:


get_ipython().run_line_magic('timeit', 'planck_spectrum(wavelength)')


# In[ ]:


solar = planck_spectrum(wavelength)


# In[ ]:


solar.flags


# #### Function calls
# 
# Conventional implementation of trapezoidal rule with explcit loop and single-valued function calls:

# In[ ]:


def integr_trapez(f, a, b, n):
    
    # integration step
    h = (b - a)/n
    
    # initialisation
    tmp = 0.5*f(a)
    
    # loop over subintervals between a+h and b-h
    for i in range(1,n):
        tmp += f(a + i*h)
        
    tmp += 0.5*f(b)
    
    return h*tmp


# In[ ]:


# verification
integr_trapez(math.sin, 0, math.pi/2, 20)


# In[ ]:


get_ipython().run_line_magic('timeit', 'integr_trapez(planck_spectrum, 1e-9, 364.7e-9, 100)')


# Implementation using Numpy arrays

# In[ ]:


import numpy as np

def integr_trapez(f, a, b, n):

    # integration step
    h = (b - a)/n
    
    # endpoints of subintervals between a+h and b-h
    x = np.linspace(a+h, b-h, n-1)
    
    return 0.5*h*(f(a) + 2*np.sum(f(x)) + f(b))


# In[ ]:


get_ipython().run_line_magic('timeit', 'integr_trapez(planck_spectrum, 1e-9, 364.7e-9, 100)')


# ## Cythonizing code

# In[ ]:


def rk4_step(f, t, x, dt):

    k1 = dt * f(t, x)
    k2 = dt * f(t + 0.5*dt, x + 0.5*k1)
    k3 = dt * f(t + 0.5*dt, x + 0.5*k2)
    k4 = dt * f(t + dt, x + k3) 

    return x + (k1 + 2*(k2 + k3) + k4)/6


# In[ ]:


import numpy as np

def solve_stroemgren(r0, dt, n_steps):
    t = np.linspace(0, n_steps*dt, n_steps+1)
    r = np.zeros(n_steps+1)
    r[0] = r0

    for n in range(n_steps):
        r[n+1] = rk4_step(lambda t, r: (1 - r**3)/(3*r**2),
                          t[n], r[n], dt)

    return (t,r)


# In[ ]:


get_ipython().run_line_magic('timeit', 'solve_stroemgren(0.01, 1e-3, 10000)')


# You need to run
# 
# ```python setup.py build_ext --inplace```
# 
# on the command line to create the C-extension module stroemgren

# In[ ]:


import numpy as np
from stroemgren import crk4_step


# In[ ]:


def solve_stroemgren(r0, dt, n_steps):
    t = np.linspace(0, n_steps*dt, n_steps+1)
    r = np.zeros(n_steps+1)
    r[0] = r0

    for n in range(n_steps):
        r[n+1] = crk4_step(lambda t, r: (1 - r**3)/(3*r**2),
                           t[n], r[n], dt)

    return (t,r)


# In[ ]:


get_ipython().run_line_magic('timeit', 'solve_stroemgren(0.01, 1e-3, 10000)')


# In[ ]:


from stroemgren import stroemgren_step


# In[ ]:


def solve_stroemgren(r0, dt, n_steps):
    t = np.linspace(0, n_steps*dt, n_steps+1)
    r = np.zeros(n_steps+1)
    r[0] = r0

    for n in range(n_steps):
        r[n+1] = stroemgren_step(t[n], r[n], dt)

    return (t,r)


# In[ ]:


get_ipython().run_line_magic('timeit', 'solve_stroemgren(0.01, 1e-3, 10000)')


# In[ ]:




