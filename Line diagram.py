import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
models = ['Baseline RF', 'Baseline XGB', 'Baseline CatBoost', 'Baseline Voting',
          'Enhanced RF', 'Enhanced XGB', 'Enhanced CatBoost', 'Enhanced Voting',
          'XAI+Enhanced RF', 'XAI+Enhanced XGB', 'XAI+Enhanced CatBoost', 'XAI+Enhanced Voting']

# Baseline model metrics
baseline_accuracy = [91.50, 92.25, 92.25, 92.75]
baseline_precision = [91.50, 92.25, 92.25, 93.00]
baseline_recall = [91.50, 92.25, 92.25, 93.00]
baseline_f1_score = [91.50, 92.25, 92.25, 93.00]

# Enhanced model metrics
enhanced_accuracy = [93.00, 93.50, 94.00, 94.25]
enhanced_precision = [93.00, 93.50, 94.00, 94.50]
enhanced_recall = [93.00, 93.50, 94.00, 94.50]
enhanced_f1_score = [93.00, 93.50, 94.00, 94.50]

# XAI + Hyperparameter tuned model metrics
xai_accuracy = [93.75, 93.25, 94.50, 94.75]
xai_precision = [93.75, 93.25, 94.50, 95.00]
xai_recall = [93.75, 93.25, 94.50, 95.00]
xai_f1_score = [93.75, 93.25, 94.50, 95.00]

# Bar positions
bar_width = 0.2
r1 = np.arange(len(models))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]

# Plotting the data
plt.figure(figsize=(20, 10))
plt.bar(r1, baseline_accuracy + enhanced_accuracy + xai_accuracy, color='cyan', width=bar_width, edgecolor='black', label='Accuracy (%)')
plt.bar(r2, baseline_precision + enhanced_precision + xai_precision, color='green', width=bar_width, edgecolor='black', label='Precision (%)')
plt.bar(r3, baseline_recall + enhanced_recall + xai_recall, color='red', width=bar_width, edgecolor='black', label='Recall (%)')
plt.bar(r4, baseline_f1_score + enhanced_f1_score + xai_f1_score, color='lightsalmon', width=bar_width, edgecolor='black', label='F1-Score (%)')

# Adding titles and labels
plt.title('Performance Metrics for RF, XGB, CatBoost, and Voting Classifiers', fontsize=30)
plt.xlabel('Model', fontsize=20)
plt.ylabel('Percentage (%)', fontsize=20)
plt.xticks([r + bar_width*1.5 for r in range(len(models))], models, rotation=90)
plt.ylim(90, 100)
plt.legend()

# Tight layout for better spacing
plt.tight_layout()

# Display the plot
plt.show()
