import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np

# Load and prepare data
df = pd.read_csv("COMBO2.csv")
drop_columns = ['Filename', "is_stroke_face"]
X = df.drop(drop_columns, axis=1)
y = df["is_stroke_face"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=150)

# 1. XGBoost Feature Importance
best_params_xgb = {
    'max_depth': 9,
    'min_child_weight': 1,
    'learning_rate': 0.2,
    'subsample': 0.8,
    'colsample_bytree': 1.0,
    'gamma': 0,
    'n_estimators': 200,
    'use_label_encoder': False,
    'eval_metric': 'rmse',
    'objective': 'binary:logistic'
}

# Train the final XGBoost model
best_xgb = xgb.XGBClassifier(**best_params_xgb)
best_xgb.fit(X_train, y_train)

# Get feature importances for XGBoost
xgb_importances = best_xgb.feature_importances_
xgb_indices = np.argsort(xgb_importances)[::-1]
xgb_feature_ranks = {X.columns[i]: xgb_importances[i] for i in xgb_indices}

# Save feature importances for XGBoost to a file
with open("xgb_feature_importances.txt", "w") as f:
    f.write("XGBoost Feature Importances:\n")
    for feature, importance in xgb_feature_ranks.items():
        f.write(f"{feature}: {importance:.4f}\n")

# Print feature importances for XGBoost
print("XGBoost Feature Importances:")
for feature, importance in xgb_feature_ranks.items():
    print(f"{feature}: {importance:.4f}")

# 2. Random Forest Feature Importance
best_params_rf = {
    'n_estimators': 300,
    'max_depth': 90,
    'min_samples_split': 6,
    'min_samples_leaf': 3,
    'max_features': 'sqrt',
    'bootstrap': False,
    'criterion': 'entropy',
    'random_state': 150
}

# Train the final Random Forest model
best_rf = RandomForestClassifier(**best_params_rf)
best_rf.fit(X_train, y_train)

# Get feature importances for Random Forest
rf_importances = best_rf.feature_importances_
rf_indices = np.argsort(rf_importances)[::-1]
rf_feature_ranks = {X.columns[i]: rf_importances[i] for i in rf_indices}

# Save feature importances for Random Forest to a file
with open("rf_feature_importances.txt", "w") as f:
    f.write("Random Forest Feature Importances:\n")
    for feature, importance in rf_feature_ranks.items():
        f.write(f"{feature}: {importance:.4f}\n")

# Print feature importances for Random Forest
print("Random Forest Feature Importances:")
for feature, importance in rf_feature_ranks.items():
    print(f"{feature}: {importance:.4f}")

# Combine the importances into a single DataFrame for comparison
features = X.columns
importance_df = pd.DataFrame({
    'Feature': features,
    'XGBoost Importance': xgb_importances,
    'Random Forest Importance': rf_importances
})

# Sort the DataFrame by XGBoost Importance for better comparison
importance_df = importance_df.sort_values(by='XGBoost Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(14, 10))
sns.barplot(x='XGBoost Importance', y='Feature', data=importance_df, color="blue", label="XGBoost")
sns.barplot(x='Random Forest Importance', y='Feature', data=importance_df, color="green", label="Random Forest", alpha=0.7)

plt.title('Feature Importance Comparison between XGBoost and Random Forest', fontsize=18)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.legend(loc='upper right')
plt.show()

# Save the combined importance plot
plt.savefig("feature_importance_comparison.png", bbox_inches='tight')
