import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Generate the impulse and the time-axis
xx = np.zeros(20)
xx[5] = 1
tt = np.arange(20)

# Put the results into a Python-dictionary
data = {}
data["before"] = xx
data["after_fir"] = signal.lfilter(np.ones(5) / 5, 1, xx)
data["after_iir"] = signal.lfilter([1], [1, -0.5], xx)

# Show the results
_, ax = plt.subplots(figsize=(6, 4))
ax.plot(tt, data["before"], "o", label="input", lw=2)
ax.plot(tt, data["after_fir"], "x-", label="FIR-filtered", lw=2)
ax.plot(tt, data["after_iir"], ".:", label="IIR-filtered", lw=2)
ax.set(xlabel="Timesteps", ylabel="Signal", xticks=np.arange(0, 20, 2))
ax.legend()
plt.show()
