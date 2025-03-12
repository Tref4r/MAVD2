import pandas as pd
import matplotlib.pyplot as plt

# Đọc file CSV
df = pd.read_csv("./training_history.csv")

# Tạo figure với kích thước lớn hơn
fig, ax1 = plt.subplots(figsize=(14, 6))
# Adding secondary y-axis for Accuracy (m_ap * 100)
ax2 = ax1.twinx()
ax2.plot(df['iteration'], df['m_ap'] * 100, label='Accuracy', linestyle='solid', color='purple')
ax2.set_ylabel('Accuracy (%)')
# Subplot 2: Loss values with m_ap accuracy as secondary y-axis
ax1.plot(df['iteration'], df['U_MIL_loss'], label='U_MIL_loss')
ax1.plot(df['iteration'], df['MA_loss'], label='MA_loss')
ax1.plot(df['iteration'], df['M_MIL_loss'], label='M_MIL_loss')
ax1.plot(df['iteration'], df['Triplet_loss'], label='Triplet_loss')

# Adding learning rate plot
ax3 = ax1.twinx()
ax3.plot(df['iteration'], df['LR'], label='Learning Rate', linestyle='dashed', color='green')
ax3.set_ylabel('Learning Rate')
ax3.spines['right'].set_position(('outward', 60))

ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss')

# Merge legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
ax1.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='upper left', bbox_to_anchor=(1.05, 1))

# Adjust layout to avoid overlap
plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1)

plt.show()