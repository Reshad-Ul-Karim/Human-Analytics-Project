import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import svm
import joblib

# Load the dataset
df = pd.read_csv("Final Combined.csv")  # Replace 'your_dataset.csv' with your actual file path

# Split the dataset into features (X) and labels (y)
X = df.drop(['Filename', "is_stroke_face"], axis=1)
y = df["is_stroke_face"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=150)
# SVM Training
model = svm.SVC(kernel='linear', C=.1, random_state=150, gamma=.5)
model.fit(X_train, y_train)
# SVM Testing
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Model Accuracy: {accuracy * 100:.2f}% SVM Classifier")
print(classification_report(y_test, y_pred, zero_division=1))
# Optional: Review Best Parameters
print("Best Parameters:", model.get_params())
# save the model
joblib.dump(model, 'SVM_Model.pkl')
print("Model Saved")
