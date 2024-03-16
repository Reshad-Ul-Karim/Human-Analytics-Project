import pandas as pd
from sklearn.model_selection import train_test_split
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
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Model Accuracy: {accuracy * 100:.2f}% Random Forest Classifier")
print(classification_report(y_test, y_pred, zero_division=1))
# confusion matrix
conf = confusion_matrix(y_test, y_pred)
# pl.matshow(conf)
# pl.title('Confusion matrix of the classifier')
# pl.colorbar()
# pl.show()
labels = ["Non-Stroke", "Stroke", "Non-Stroke", "Stroke"]
group_percentages = ["{0:.2%}".format(value) for value in
                     conf.flatten() / np.sum(conf)]
categories = ["Non_stroke", "Stroke"]
labels = np.asarray(labels).reshape(2, 2)
group_percentages = np.asarray(group_percentages).reshape(2, 2)
sns.heatmap(conf, annot=group_percentages, fmt="", cmap="crest", xticklabels=categories, yticklabels=categories)
plt.show()
# save the model
if accuracy >= 0.95:
    joblib.dump(rf, 'Random_Forest_Model.pkl')
    print("Model Saved")
