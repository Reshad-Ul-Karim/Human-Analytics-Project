import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the dataset
df = pd.read_csv("Final Combined.csv")  # Replace 'your_dataset.csv' with your actual file path

# Split the dataset into features (X) and labels (y)
X = df.drop(['Filename', "is_stroke_face"], axis=1)

y = df["is_stroke_face"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=150)
# Hyperparameter tuning setup
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
}


rf = RandomForestClassifier(random_state=150)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

grid_search.fit(X_train, y_train)

# Select the best model
best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Improved Model Accuracy: {accuracy * 100:.2f}% using Random Forest Classifier")
print(classification_report(y_test, y_pred, zero_division=1))



'''=========================================================================='''
# Confusion matrix visualization
conf = confusion_matrix(y_test, y_pred)
sns.heatmap(conf, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Stroke", "Stroke"], yticklabels=["Non-Stroke", "Stroke"])
plt.title('Confusion Matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

'''=========================================================================='''
# confusion matrix
conf = confusion_matrix(y_test, y_pred)
# pl.matshow(conf)
# pl.title('Confusion matrix of the classifier')
# pl.colorbar()
# pl.show()

'''
labels = ["Non-Stroke", "Stroke", "Non-Stroke", "Stroke"]
group_percentages = ["{0:.2%}".format(value) for value in
                     conf.flatten() / np.sum(conf)]
categories = ["Non_stroke", "Stroke"]
labels = np.asarray(labels).reshape(2, 2)
group_percentages = np.asarray(group_percentages).reshape(2, 2)
sns.heatmap(conf, annot=group_percentages, fmt="", cmap="crest", xticklabels=categories, yticklabels=categories)
plt.show()
'''

'''=========================================================================='''

# save the model
if accuracy >= 0.95:
    joblib.dump(rf, 'Random_Forest_Model.pkl')
    print("Model Saved")
