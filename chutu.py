import matplotlib.pyplot as plt
import numpy as np

# Data from the image
categories = ['original', 'first_migration', 'second_migration']
diameters = [186.02, 8.27, 4.13]

energies = [13362.82, 10868.37, 10643.07]

# Set up the figure and axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar plot for the diameters
ax1.bar(categories, diameters, width=0.2, label='Average Diameter', align='center', color='royalblue', alpha=0.7)

# Create a second y-axis for fractal dimensions and energy
ax2 = ax1.twinx()

# Line plot for fractal dimensions

# Line plot for energies
ax2.plot(categories, energies, marker='o', label='Energy', color='red', linestyle='-', linewidth=2)

# Set labels and title
ax1.set_xlabel('Categories')
ax1.set_ylabel('Average Diameter', color='royalblue')
ax2.set_ylabel('Fractal Dimension & Energy', color='black')

# Add legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Show the plot
plt.title('Comparison of Topological Features for Different Categories')
plt.show()
