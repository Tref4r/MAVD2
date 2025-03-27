import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc

# Load the CSV file
df = pd.read_csv('training_history.csv')

# Extract the relevant columns
# Assuming the CSV has columns 'iteration' and 'm_ap'
iterations = df['iteration']
ap = df['m_ap'] * 100  # Convert mAP to AP

# Calculate AUC
area_under_curve = auc(iterations, ap)

# Plot the AUC
plt.figure(figsize=(18, 10))
plt.plot(iterations, ap, label=f'AUC = {area_under_curve:.2f}', marker='o')
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('AP (%)', fontsize=14)
plt.title('Area Under the Curve (AUC) of AP over Iterations', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
