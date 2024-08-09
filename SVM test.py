from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
}
df = pd.read_csv("COMBO.csv")  # Replace 'your_dataset.csv' with your actual file path

# Split the dataset into features (X) and labels (y)
X = df.drop(['Filename', "is_stroke_face"], axis=1)

y = df["is_stroke_face"]
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=150, )
# Create a base model
svm_model = svm.SVC()

# Instantiate the random search model
random_search = RandomizedSearchCV(estimator=svm_model, param_distributions=param_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)

# Fit the random search model
random_search.fit(X_train, y_train)

# Get the best parameters
best_params = random_search.best_params_
print(f"Best parameters: {best_params}")

# Train the model using the best parameters
optimized_svm = svm.SVC(**best_params)
optimized_svm.fit(X_train, y_train)

# Predict the test set results
y_pred = optimized_svm.predict(X_test)

# Print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy for optimized SVM: {accuracy * 100:.2f}%')