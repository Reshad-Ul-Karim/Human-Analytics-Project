import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the datasets
df_straight = pd.read_csv('flattened_straight face_landmarks.csv')
df_droopy = pd.read_csv('flattened_droopy_landmarks.csv')

# Label the data
df_straight['label'] = 0
df_droopy['label'] = 1

# Combine the datasets
df = pd.concat([df_straight, df_droopy], ignore_index=True)

# Optional: Apply feature engineering techniques here

# Shuffle the combined dataset
df = df.sample(frac=1).reset_index(drop=True)

# Exclude the 'Filename' column and split the dataset
X = df.drop(['label', 'Filename'], axis=1)
y = df['label']

from sklearn.impute import SimpleImputer

# Imputing numeric columns with the median
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

X_dropped = X.dropna()
y_dropped = y[X.index.isin(X_dropped.index)]

from imblearn.over_sampling import SMOTE

# Assuming X_imputed is your dataset after handling missing values
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_imputed, y)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Hyperparameter tuning setup
param_dist_rf = {
    'n_estimators': randint(100, 1200),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 15),
    'min_samples_leaf': randint(1, 15),
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy'],
    'max_features': ['auto', 'sqrt', 'log2', None]
}

# Initialize the RandomForestClassifier
model_rf = RandomForestClassifier(random_state=42)

# Setup RandomizedSearchCV
random_search_rf = RandomizedSearchCV(
    estimator=model_rf,
    param_distributions=param_dist_rf,
    n_iter=100,
    scoring='accuracy',
    cv=10,
    random_state=42,
    verbose=3
)

# Perform hyperparameter tuning
random_search_rf.fit(X_train, y_train)

# Training the best model (Optional: as best estimator is already fitted)
best_model_rf = random_search_rf.best_estimator_

# Evaluating the model
y_pred_rf = best_model_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Best RandomForest Model Accuracy: {accuracy_rf * 100:.2f}%")
print(classification_report(y_test, y_pred_rf, zero_division=1))

# Review best parameters
print("Best Parameters for RandomForest:", random_search_rf.best_params_)
print("Best RandomForest Estimator:", random_search_rf.best_estimator_)
