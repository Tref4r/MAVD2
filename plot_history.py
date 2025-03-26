import pandas as pd
import matplotlib.pyplot as plt

# Đọc file CSV
df = pd.read_csv("./training_history.csv")

# Tạo figure với kích thước lớn hơn
fig, ax1 = plt.subplots(figsize=(18, 10))  # Increase the figure size
# Adding secondary y-axis for Accuracy (AP)
ax2 = ax1.twinx()
ax2.plot(df['iteration'], df['m_ap'] * 100, label='Accuracy', linestyle='solid', color='purple')
ax2.set_ylabel('Accuracy', fontsize=14)
# Subplot 2: Loss values with AP accuracy as secondary y-axis
ax1.plot(df['iteration'], df['U_MIL_loss'], label='U_MIL_loss')
ax1.plot(df['iteration'], df['MA_loss'], label='MA_loss')
ax1.plot(df['iteration'], df['M_MIL_loss'], label='M_MIL_loss')
ax1.plot(df['iteration'], df['Triplet_loss'], label='Triplet_loss')

ax1.set_xlabel('Iteration', fontsize=14)
ax1.set_ylabel('Loss', fontsize=14)

# Merge legends
ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)
ax2.legend(loc='lower left', bbox_to_anchor=(1.05, 0), fontsize=12)

# Adjust layout to avoid overlap
plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
