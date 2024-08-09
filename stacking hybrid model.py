# ... existing imports ...
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import StackingClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
import pandas as pd

df = pd.read_csv('COMBO2.csv')
drop_columns = ['Filename', "is_stroke_face"]
ranked_regions = ['rightCheek', 'leftCheek', 'lipsUpperOuter', 'lipsUpperInner', 'lipsLowerInner', 'lipsLowerOuter',
                  'rightEyebrowUpper', 'rightEyeLower3', 'rightEyeLower2', 'leftEyeLower3', 'leftEyebrowUpper',
                  'rightEyeLower1', 'leftEyeLower2', 'rightEyeLower0', 'leftEyeLower1', 'leftEyeLower0',
                  'rightEyeUpper2', 'leftEyeUpper2', 'leftEyeUpper1', 'rightEyeUpper1', 'leftEyebrowLower',
                  'rightEyebrowLower', 'rightEyeUpper0', 'leftEyeUpper0', 'noseRightCorner', 'noseLeftCorner',
                  'noseBottom', 'midwayBetweenEyes']
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
    "noseBottom": [2],
    "noseRightCorner": [98],
    "noseLeftCorner": [327],
    "rightCheek": [205, 137, 123, 50, 203, 177, 147, 187, 207, 216, 215, 213, 192, 214, 212, 138, 135, 210, 169],
    "leftCheek": [425, 352, 280, 330, 266, 423, 426, 427, 411, 376, 436, 416, 432, 434, 422, 430, 364, 394, 371]
}
c = 0
for region in ranked_regions:
    if c >= 3:
        for index in FACE_INDEXES[region]:
            drop_columns.append(f"{region}_{index}")
    c += 1
X = df.drop(drop_columns, axis=1)
y = df["is_stroke_face"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

hyperparameters_RFC = {'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2,
                       'n_estimators': 200,
                       }

hyperparameters_XGB = {'subsample': 0.7000000000000001, 'n_estimators': 200, 'max_depth': None, 'learning_rate': '0.05',
                       'gamma': 0.5, 'colsample_bytree': 0.6000000000000001, 'use_label_encoder': False,
                       'eval_metric': 'rmse', 'min_child_weight': 2, 'objective': 'binary:logistic',
                       }
hyperparameters_CB = {'bagging_temperature': 0.8607305832563434, 'bootstrap_type': 'MVS',
                      'colsample_bylevel': 0.917411003148779,
                      'depth': 8, 'grow_policy': 'SymmetricTree', 'iterations': 918, 'l2_leaf_reg': 8,
                      'learning_rate': 0.29287291117375575, 'max_bin': 231, 'min_data_in_leaf': 9, 'od_type': 'Iter',
                      'od_wait': 21, 'one_hot_max_size': 7, 'random_strength': 0.6963042728397884,
                      'scale_pos_weight': 1.924541179848884, 'subsample': 0.6480869299533999}
# Combine all parameter distributions
cb = CatBoostClassifier(**hyperparameters_CB)
rf = RandomForestClassifier(**hyperparameters_RFC)
xgb = xgb.XGBClassifier(**hyperparameters_XGB)
# Define the parameter distribution for the StackingClassifier
param_dist = {
    'cv': [None] + list(range(2, 10)),  # None and random integer from 2 to 10
    'stack_method': ['auto', 'predict_proba', 'decision_function', 'predict', None],
    'n_jobs': [None] + list(range(-1, 10)),  # None and random integer from -1 to 10
    'passthrough': [True, False, None],
    'verbose': [None] + list(range(0, 10))  # None and random integer from 0 to 10
}

# Initialize the StackingClassifier
stacking_clf = StackingClassifier(
    estimators=[
        ('catboost', cb),
        ('rf', rf),
        ('xgb', xgb)
    ],
    final_estimator=cb,
    cv=5
)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=stacking_clf,
    param_distributions=param_dist,
    n_iter=100,  # Number of parameter settings sampled
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=150,
    verbose=2
)

# Perform random search
random_search.fit(X_train, y_train)

# Print best parameters and score
print("Best parameters:", random_search.best_params_)
print("Best cross-validation score:", random_search.best_score_)

# Evaluate the best model on the test set
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test set accuracy: {accuracy:.2f}')
