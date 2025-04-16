import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ['Original', 'Universal Style Transfer', 'First Migration', 'Second Migration']
diameters = [70.357, 56.675, 53.850, 47.933]  # Network Diameter

# Set up the figure and axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar plot for the diameters
ax1.bar(categories, diameters, width=0.3, label='Network Diameter', align='center', color='royalblue', alpha=0.7)




# Set labels and title
ax1.set_xlabel('Categories')
ax1.set_ylabel('Network Diameter', color='royalblue')

# Add legends
ax1.legend(loc='upper left')


# Display the plot
plt.title('Comparison of Topological Features for Different Categories')
plt.show()
