import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv("COMBO2.csv")
drop_columns = ['Filename', "is_stroke_face"]
X = df.drop(drop_columns, axis=1)
y = df["is_stroke_face"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=150)

# Updated best hyperparameters provided
best_params = {
    'n_estimators': 300,
    'max_depth': 90,
    'min_samples_split': 6,
    'min_samples_leaf': 3,
    'max_features': 'sqrt',
    'bootstrap': False,
    'criterion': 'entropy',
    'random_state': 150
}

# Train the final model with the updated best hyperparameters
best_rf = RandomForestClassifier(**best_params)
best_rf.fit(X_train, y_train)

# Make predictions
y_pred = best_rf.predict(X_test)

# Open a file to save the results
with open("classification_results_RFC.txt", "w") as f:
    # Generate classification report
    report = classification_report(y_test, y_pred)
    f.write("Classification Report:\n")
    f.write(report + "\n\n")

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    f.write("Confusion Matrix:\n")
    f.write(str(conf_matrix) + "\n\n")

    # Calculate FP, FN, TP, TN percentages
    tn, fp, fn, tp = conf_matrix.ravel()
    total = tn + fp + fn + tp

    tn_percentage = tn / total * 100
    fp_percentage = fp / total * 100
    fn_percentage = fn / total * 100
    tp_percentage = tp / total * 100

    # Write percentages to the file
    f.write(f"True Negatives (TN): {tn_percentage:.2f}%\n")
    f.write(f"False Positives (FP): {fp_percentage:.2f}%\n")
    f.write(f"False Negatives (FN): {fn_percentage:.2f}%\n")
    f.write(f"True Positives (TP): {tp_percentage:.2f}%\n\n")

    # Write percentage table to the file
    percentage_data = {
        "Category": ["True Negatives (TN)", "False Positives (FP)", "False Negatives (FN)", "True Positives (TP)"],
        "Percentage": [tn_percentage, fp_percentage, fn_percentage, tp_percentage]
    }
    percentage_df = pd.DataFrame(percentage_data)
    f.write("Percentage Table:\n")
    f.write(percentage_df.to_string(index=False) + "\n")

# Plot confusion matrix with extra-large fonts
plt.figure(figsize=(12, 9))  # Adjust the figure size for better visibility
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Stroke', 'Stroke'], yticklabels=['Non-Stroke', 'Stroke'],
            annot_kws={"size": 70})  # Increase font size for annotations
plt.xlabel('Predicted Label', fontsize=30)  # Increase font size for xlabel
plt.ylabel('True Label', fontsize=30)  # Increase font size for ylabel
plt.title('Confusion Matrix', fontsize=35)  # Increase font size for title
plt.xticks(fontsize=25)  # Increase font size for x-ticks
plt.yticks(fontsize=25)  # Increase font size for y-ticks

# Use tight_layout to minimize extra whitespace
plt.tight_layout()

# Save the figure with reduced whitespace
plt.savefig("confusion_matrix_RFC.png", bbox_inches='tight')

plt.show()
