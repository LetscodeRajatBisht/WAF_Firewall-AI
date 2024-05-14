import matplotlib.pyplot as plt
import numpy as np

# Define classifiers, F1-scores, accuracies, and ROC AUCs
classifiers = ['Logistic Regression', 'Decision Tree', 'Linear SVC', 'Random Forest']
f1_scores = [0.9788, 0.9980, 0.9932, 0.9966]  # F1-scores
accuracies = [0.9985, 0.9999, 0.9995, 0.9998]  # Accuracies
roc_auc_scores = [0.9810, 0.9987, 0.9946, 0.9974]  # ROC AUCs

# Set width of bar
bar_width = 0.2

# Set position of bar on X axis
r1 = np.arange(len(classifiers))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Plot the grouped bar chart with enhanced aesthetics
plt.figure(figsize=(10, 6))

# Plot bars for F1-score
plt.bar(r1, f1_scores, color='#1f77b4', width=bar_width, edgecolor='grey', label='F1-score')
# Plot bars for Accuracy
plt.bar(r2, accuracies, color='#ff7f0e', width=bar_width, edgecolor='grey', label='Accuracy')
# Plot bars for ROC AUC
plt.bar(r3, roc_auc_scores, color='#2ca02c', width=bar_width, edgecolor='grey', label='ROC AUC')

# Add xticks on the middle of the group bars
plt.xlabel('Classifier', fontweight='bold', fontsize=12)
plt.xticks([r + bar_width for r in range(len(classifiers))], classifiers, rotation=45, ha='right', fontsize=10)

# Set y-axis range based on the maximum and minimum values
min_value = min(min(f1_scores), min(accuracies), min(roc_auc_scores))
max_value = max(max(f1_scores), max(accuracies), max(roc_auc_scores))
plt.ylim(min_value - 0.01, max_value + 0.01)

# Add labels and legend
plt.ylabel('Scores', fontweight='bold', fontsize=12)
plt.title('Performance of Classifiers', fontweight='bold', fontsize=14)
plt.legend()

# Add grid lines
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show plot
plt.tight_layout()
plt.show()
