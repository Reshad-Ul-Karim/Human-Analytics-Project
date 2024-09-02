import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
# Assuming the provided FACE_INDEXES and ranked_regions are already defined as given in the prompt.
FACE_INDEXES = {
    "lipsUpperOuter": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
    "lipsLowerOuter": [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    "lipsUpperInner": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
    "lipsLowerInner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
    "rightEyeUpper0": [246, 161, 160, 159, 158, 157, 173],
    "rightEyeLower0": [33, 7, 163, 144, 145, 153, 154, 155, 133],
    "rightEyeUpper1": [247, 30, 29, 27, 28, 56, 190],
    "rightEyeLower1": [130, 25, 110, 24, 23, 22, 26, 112, 243],
    "rightEyeUpper2": [113, 225, 224, 223, 222, 221, 189],
    "rightEyeLower2": [226, 31, 228, 229, 230, 231, 232, 233, 244],
    "rightEyeLower3": [143, 111, 117, 118, 119, 120, 121, 128, 245],
    "rightEyebrowUpper": [156, 70, 63, 105, 66, 107, 55, 193],
    "rightEyebrowLower": [35, 124, 46, 53, 52, 65],
    "leftEyeUpper0": [466, 388, 387, 386, 385, 384, 398],
    "leftEyeLower0": [263, 249, 390, 373, 374, 380, 381, 382, 362],
    "leftEyeUpper1": [467, 260, 259, 257, 258, 286, 414],
    "leftEyeLower1": [359, 255, 339, 254, 253, 252, 256, 341, 463],
    "leftEyeUpper2": [342, 445, 444, 443, 442, 441, 413],
    "leftEyeLower2": [446, 261, 448, 449, 450, 451, 452, 453, 464],
    "leftEyeLower3": [372, 340, 346, 347, 348, 349, 350, 357, 465],
    "leftEyebrowUpper": [383, 300, 293, 334, 296, 336, 285, 417],
    "leftEyebrowLower": [265, 353, 276, 283, 282, 295],
    "midwayBetweenEyes": [168],
    "noseTip": [1],
    "noseBottom": [2],
    "noseRightCorner": [98],
    "noseLeftCorner": [327],
    "rightCheek": [205, 137, 123, 50, 203, 177, 147, 187, 207, 216, 215, 213, 192, 214, 212, 138, 135, 210, 169],
    "leftCheek": [425, 352, 280, 330, 266, 423, 426, 427, 411, 376, 436, 416, 432, 434, 422, 430, 364, 394, 371]
}

ranked_regions = ['rightCheek', 'leftCheek', 'lipsUpperOuter', 'lipsUpperInner', 'lipsLowerInner', 'lipsLowerOuter',
                  'rightEyebrowUpper', 'rightEyeLower3', 'rightEyeLower2', 'leftEyeLower3', 'leftEyebrowUpper',
                  'rightEyeLower1', 'leftEyeLower2', 'rightEyeLower0', 'leftEyeLower1', 'leftEyeLower0',
                  'rightEyeUpper2', 'leftEyeUpper2', 'leftEyeUpper1', 'rightEyeUpper1', 'leftEyebrowLower',
                  'rightEyebrowLower', 'rightEyeUpper0', 'leftEyeUpper0', 'noseRightCorner', 'noseLeftCorner',
                  'noseBottom', 'midwayBetweenEyes']
# Load and prepare data
df = pd.read_csv("COMBO2.csv")
drop_columns = ['Filename', "is_stroke_face"]
X = df.drop(drop_columns, axis=1)
y = df["is_stroke_face"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=150)

# Train the XGBoost model
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
best_xgb = xgb.XGBClassifier(**best_params_xgb)
best_xgb.fit(X_train, y_train)
xgb_importances = best_xgb.feature_importances_

# Train the Random Forest model
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
best_rf = RandomForestClassifier(**best_params_rf)
best_rf.fit(X_train, y_train)
rf_importances = best_rf.feature_importances_

def calculate_region_importance(importances, face_indexes):
    region_importances = {}
    for region, indexes in face_indexes.items():
        region_importances[region] = np.sum([importances[i] for i in indexes if i < len(importances)])
    return region_importances

# Calculate region importances for XGBoost
xgb_region_importances = calculate_region_importance(xgb_importances, FACE_INDEXES)

# Calculate region importances for Random Forest
rf_region_importances = calculate_region_importance(rf_importances, FACE_INDEXES)

# Rank regions by importance for both models
xgb_sorted_regions = sorted(xgb_region_importances.items(), key=lambda item: item[1], reverse=True)
rf_sorted_regions = sorted(rf_region_importances.items(), key=lambda item: item[1], reverse=True)

# Print the ranked regions for XGBoost
print("XGBoost Ranked Regions by Importance:")
for region, importance in xgb_sorted_regions:
    print(f"{region}: {importance:.4f}")

# Print the ranked regions for Random Forest
print("Random Forest Ranked Regions by Importance:")
for region, importance in rf_sorted_regions:
    print(f"{region}: {importance:.4f}")

with open("ranked_region_importances.txt", "w") as f:
    # Write XGBoost ranked regions
    f.write("XGBoost Ranked Regions by Importance:\n")
    for region, importance in xgb_sorted_regions:
        f.write(f"{region}: {importance:.4f}\n")
    f.write("\n")

    # Write Random Forest ranked regions
    f.write("Random Forest Ranked Regions by Importance:\n")
    for region, importance in rf_sorted_regions:
        f.write(f"{region}: {importance:.4f}\n")
    f.write("\n")


# Combine the importances into a single DataFrame for comparison
importance_df = pd.DataFrame({
    'Region': [region for region, _ in xgb_sorted_regions],
    'XGBoost Importance': [importance for _, importance in xgb_sorted_regions],
    'Random Forest Importance': [rf_region_importances[region] for region, _ in xgb_sorted_regions]
})

# Plot the feature importances
plt.figure(figsize=(14, 10))
bar_width = 0.4
index = np.arange(len(importance_df))

plt.bar(index, importance_df['XGBoost Importance'], bar_width, color='blue', label='XGBoost')
plt.bar(index + bar_width, importance_df['Random Forest Importance'], bar_width, color='green', label='Random Forest')

plt.xlabel('Region', fontsize=14)
plt.ylabel('Importance', fontsize=14)
plt.title('Region Importance Comparison between XGBoost and Random Forest', fontsize=18)
plt.xticks(index + bar_width / 2, importance_df['Region'], rotation=90)
plt.legend()

plt.tight_layout()
plt.show()