import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

# Set background color
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Remove axes
ax.axis('off')

# Add title with custom styling
ax.text(0.5, 0.65, 'LongevAI', fontsize=60, ha='center', va='center', 
        fontweight='bold', color='#2E5EAA')

# Add subtitle
ax.text(0.5, 0.4, 'Transformer-Based Forecasting for', fontsize=16, 
        ha='center', va='center', color='#555555')
ax.text(0.5, 0.3, 'Global Life Expectancy', fontsize=16, 
        ha='center', va='center', color='#555555')

# Add a decorative wave (representing life expectancy trends)
x = np.linspace(0.1, 0.9, 100)
y = 0.8 + 0.05 * np.sin(x * 30)
ax.plot(x, y, color='#6AC7B0', linewidth=3)

# Add circles representing data points
ax.scatter([0.3, 0.5, 0.7], [0.78, 0.83, 0.79], color='#4F8FF7', s=100, zorder=5)

# Add a semi-transparent box for a clean look
rect = patches.Rectangle((0.05, 0.2), 0.9, 0.7, linewidth=1, 
                         edgecolor='#E0E0E0', facecolor='none', 
                         alpha=0.5, zorder=-1)
ax.add_patch(rect)

# Save the figure with tight layout
plt.tight_layout()
plt.savefig('images/longevai_logo.png', dpi=300, bbox_inches='tight')
plt.close() 