import pandas as pd
import matplotlib.pyplot as plt

# Đọc file CSV
df = pd.read_csv('d:\\MAVD2\\training_history.csv')

# Vẽ biểu đồ
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(df['iteration'], df['m_ap'], label='m_ap')
plt.xlabel('Iteration')
plt.ylabel('m_ap')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(df['iteration'], df['U_MIL_loss'], label='U_MIL_loss')
plt.plot(df['iteration'], df['MA_loss'], label='MA_loss')
plt.plot(df['iteration'], df['M_MIL_loss'], label='M_MIL_loss')
plt.plot(df['iteration'], df['Triplet_loss'], label='Triplet_loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(df['iteration'], df['LR'], label='Learning Rate')
plt.xlabel('Iteration')
plt.ylabel('Learning Rate')
plt.legend()

plt.tight_layout()
plt.show()
