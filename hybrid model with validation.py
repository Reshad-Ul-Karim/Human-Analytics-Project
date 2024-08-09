import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load data
df = pd.read_csv('COMBO2.csv')
drop_columns = ['Filename', "is_stroke_face"]
X = df.drop(drop_columns, axis=1)
y = df["is_stroke_face"]

# Define hyperparameters
hyperparameters_RFC = {
    'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200
}
hyperparameters_XGB = {
    'subsample': 0.7, 'n_estimators': 200, 'max_depth': None, 'learning_rate': 0.05,
    'gamma': 0.5, 'colsample_bytree': 0.6, 'use_label_encoder': False,
    'eval_metric': 'rmse', 'min_child_weight': 2, 'objective': 'binary:logistic'
}

# Initialize classifiers
xgb = XGBClassifier(**hyperparameters_XGB)
rf = RandomForestClassifier(**hyperparameters_RFC)
svm = SVC(probability=True, kernel='rbf', gamma=1, C=10, random_state=42)

# Perform cross-validation for each model
cv_scores_xgb = cross_val_score(xgb, X, y, cv=5, scoring='accuracy')
cv_scores_rf = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
cv_scores_svm = cross_val_score(svm, X, y, cv=5, scoring='accuracy')

# Print cross-validation results for each model
print("XGBoost CV Scores:", cv_scores_xgb)
print("XGBoost CV Average Score: %.2f" % cv_scores_xgb.mean())

print("Random Forest CV Scores:", cv_scores_rf)
print("Random Forest CV Average Score: %.2f" % cv_scores_rf.mean())

print("SVM CV Scores:", cv_scores_svm)
print("SVM CV Average Score: %.2f" % cv_scores_svm.mean())
