import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load datasets
stroke_df = pd.read_csv('flattened_droopy_landmarks.csv')
non_stroke_df = pd.read_csv('flattened_straight face_landmarks.csv')

# Assign labels
stroke_df['label'] = 1  # Stroke
non_stroke_df['label'] = 0  # Non-stroke

# Combine datasets
combined_df = pd.concat([stroke_df, non_stroke_df], axis=0).reset_index(drop=True)

# Shuffle the dataset
combined_df = combined_df.sample(frac=1).reset_index(drop=True)

# Separate features and labels
X = combined_df.drop('label', axis=1)
y = combined_df['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"Training a Model"

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

"Hyperparameter Tuning with RandomizedSearchCV"

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

params = {
    'n_estimators': randint(100, 500),
    'learning_rate': uniform(0.01, 0.1),
    'subsample': uniform(0.3, 0.7),
    'max_depth': randint(3, 7),
    'colsample_bytree': uniform(0.5, 0.9),
    'min_child_weight': randint(1, 5)
}

random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=50, cv=3, verbose=2, n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# Best model
best_model = random_search.best_estimator_
print("Best Hyperparameters:", random_search.best_params_)

# Predict & Evaluate with the best model
y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best))

"feature importance visualisation"

import matplotlib.pyplot as plt

# Assuming best_model is your trained model
feature_importances = best_model.feature_importances_
plt.figure(figsize=(10, 8))
plt.barh(range(len(feature_importances)), feature_importances)
plt.yticks(range(len(feature_importances)), X.columns)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.show()


"Evaluation and model interpretation :  accuracy, precision, recall, F1 score, ROC-AUC score. "

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import seaborn as sns

# Accuracy and ROC-AUC Score
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_best))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Feature Importance Visualization (continued from the previous step)
# Helps in understanding which features are most important for the model
"deployment"
import joblib

# Save the model
joblib.dump(best_model, 'stroke_detection_model.pkl')

# Optionally, save the scaler
joblib.dump(scaler, 'scaler.pkl')
