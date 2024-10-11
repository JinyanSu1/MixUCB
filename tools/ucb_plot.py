import numpy as np
import matplotlib.pyplot as plt

# Example data
x = np.array([1, 2, 3, 4, 5, 6, 7])  # x-axis (e.g., actions or iterations)
means = np.array([2.5, 3.5, 1.8, 4.2, 2.9, 2.5, 5.0])  # mean values (e.g., rewards)
stds = np.array([0.3, 0.5, 0.2, 0.4, 0.3, 0.5, 0.7])  # standard deviation (std) values

# Plot the mean values with error bars representing std (as confidence intervals)
plt.figure(figsize=(8, 5))
plt.bar(x, means, yerr=stds, capsize=5, color='skyblue', edgecolor=None)

# Customize the plot
# plt.xlabel('Actions')
# plt.ylabel('Reward')
# plt.title('UCB Confidence')
# plt.xticks(x)
# plt.grid(True)
ax=plt.gca()
plt.xticks([])
plt.yticks([])
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Show the plot
plt.show()
