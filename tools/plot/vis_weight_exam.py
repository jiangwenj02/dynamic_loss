import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

# create random data
xdata = np.random.random([2, 1000])

# split the data into two parts
xdata1 = xdata[0, :]
xdata2 = xdata[1, :]

# sort the data so it makes clean curves
xdata1.sort()
xdata2.sort()

# create some y data points
ydata1 = xdata1 ** 2
ydata2 = 1 - xdata2 ** 2

xdata11 = xdata1 * 4
xdata22 = xdata2 * 100

# plot the data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# ax.plot(xdata11, ydata1, color='tab:blue', label='Class Imbalance')
# ax = fig.add_subplot(2, 1, 1)
# ax.plot(xdata22, ydata2, color='tab:orange', label='Corrpute Labels')
ax.plot(xdata22, ydata2, color='tab:orange', label='noise samples')
ax.legend()
# create the events marking the x data points
# xevents1 = EventCollection(xdata1, color='tab:blue', linelength=0.05)
# xevents2 = EventCollection(xdata2, color='tab:orange', linelength=0.05)

# create the events marking the y data points
# yevents1 = EventCollection(ydata1, color='tab:blue', linelength=0.05,
#                            orientation='vertical')
# yevents2 = EventCollection(ydata2, color='tab:orange', linelength=0.05,
#                            orientation='vertical')

# add the events to the axis
# ax.add_collection(xevents1)
# ax.add_collection(xevents2)
# ax.add_collection(yevents1)
# ax.add_collection(yevents2)

# set the limits
ax.set_xlim([0, 100])
ax.set_ylim([0, 1])

# ax.set_title('Weights Vary with Loss Values')
ax.set_title('Probability of noise with Loss Rank')
# ax.set_xlabel('Loss')
# ax.set_ylabel('Weight')
ax.set_xlabel('Rank')
ax.set_ylabel('Pro')
# display the plot
plt.show()