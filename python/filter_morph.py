import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Create the data
x = np.zeros(20)
x[10:] = 1
# Add some noise-spikes
x[[5, 15]] = 3
# Median filter the signal
x_med = signal.medfilt(x, 3)
# Average filtered data
b = np.ones(3) / 3
x_filt = signal.lfilter(b, 1, x)

# Plot the data
_, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, "o", linestyle="dotted", label="rawdata")
ax.plot(x_filt[1:], label="average")
ax.plot(x_med, label="median")
ax.set(xlim=[0, 19], xticks=np.arange(0, 20, 2))
ax.legend()
plt.show()
