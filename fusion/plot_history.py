import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("./training_history.csv")

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))  # Two subplots stacked vertically

# Subplot 1: Accuracy (AP)
ax1.plot(df['iteration'], df['m_ap'] * 100, label='Accuracy', linestyle='solid', color='purple')

ax1.set_ylabel('Accuracy (%)', fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(True)

# Subplot 2: Loss values
ax2.plot(df['iteration'], df['U_MIL_loss'], label='U_MIL_loss')
ax2.plot(df['iteration'], df['MA_loss'], label='MA_loss')
ax2.plot(df['iteration'], df['M_MIL_loss'], label='M_MIL_loss')
ax2.plot(df['iteration'], df['Contrastive_loss'], label='Contrastive_loss')  # Update key


ax2.set_xlabel('Iteration', fontsize=14)
ax2.set_ylabel('Loss', fontsize=14)
ax2.legend(fontsize=12)
ax2.grid(True)

# Adjust layout to avoid overlap
plt.tight_layout()
plt.show()
