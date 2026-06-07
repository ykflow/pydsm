import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from plotting_tools.set_plotting_theme import set_theme, colors


set_theme()

def power(tau, lmbda):
    return (1 + tau*lmbda) /lmbda


tau = np.linspace(0, 3, 100)
lmbdas = [-1, 0.1, 0.2]

fig, ax = plt.subplots(figsize=(20,10))
for lmbda in lmbdas:
    ax.plot(tau, power(tau, lmbda))

plt.tight_layout()
plt.show()