import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal

"Median filtering for 1d signal."

# Create some time signal
t = np.linspace(0,10,200)
# Create a simple sine wave
x1 = np.sin(t)
# Add noise to the signal
x2 = x1 + np.random.rand(200)
# Add noise to the signal
y1 = sp.signal.medfilt(x2,21)

# Plot the results
plt.subplot(2,1,1)
plt.plot(t,x2,'yo-')
plt.title('input wave')
plt.xlabel('time')
plt.subplot(2,1,2)
plt.plot(range(200),y1,'yo-')
plt.title('filtered wave')
plt.xlabel('time')
plt.show()
