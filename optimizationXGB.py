import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from xgboost import XGBClassifier

# Load the dataset
df = pd.read_csv("COMBO.csv")

# Split the dataset into features (X) and labels (y)
X = df.drop(['Filename', "is_stroke_face"], axis=1)
y = df["is_stroke_face"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define the hyperparameter grid for Randomized Search
random_grid = {
    'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
    'learning_rate': ['0.01', '0.05', '0.1', '0.3', '0.5'],
    'max_depth': [int(x) for x in np.linspace(3, 10, num=8)],
    'colsample_bytree': [float(x) for x in np.linspace(0.3, 0.9, num=7)],
    'subsample': [float(x) for x in np.linspace(0.1, 1, num=10)],
    'gamma': [0, 0.25, 0.5, 1.0]
}

# Define the hyperparameter grid for Grid Search
grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': ['0.01', '0.1', '0.3'],
    'max_depth': [3, 4, 5, 6],
    'colsample_bytree': [0.3, 0.5, 0.7],
    'subsample': [0.2, 0.4, 0.6, 0.8, 1],
    'gamma': [0, 0.25, 0.5]
}
# Initialize the XGBClassifier
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', )

# Perform Randomized Search
random_search = RandomizedSearchCV(estimator=xgb, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

# Print the best parameters from Randomized Search

# Perform Grid Search
grid_search = GridSearchCV(estimator=xgb, param_grid=grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Print the best parameters from Grid Search
print("Grid:", grid_search.best_params_)
print("Random:", random_search.best_params_)