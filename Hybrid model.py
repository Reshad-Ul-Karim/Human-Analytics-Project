import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit, cross_val_score
from catboost import CatBoostClassifier

# Load data
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

# x_train, x_Combine, y_train, y_Combine = train_test_split(X, y,
#                                                           train_size=0.8,
#                                                           random_state=150)
#
# # Splitting combined dataset in 50-50 fashion .i.e.
# # Testing set is 50% of combined dataset
# # Validation set is 50% of combined dataset
# x_val, x_test, y_val, y_test = train_test_split(x_Combine,
#                                                 y_Combine,
#                                                 test_size=0.5,
#                                                 random_state=150)

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
# hyperparameters_SVM = {'kernel': 'rbf', 'gamma': 1, 'C': 10, 'random_state': 42}
# hyperparameters_gbm = ({'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.11555555555555555},
#                        {'n_estimators': 250, 'max_depth': 4, 'learning_rate': 0.1577777777777778})

# xgb = XGBClassifier(**hyperparameters_XGB)
# rf = RandomForestClassifier(**hyperparameters_RFC)
# # svm = SVC(**hyperparameters_SVM)
# cb = CatBoostClassifier(**hyperparameters_CB)
# voting_clf = VotingClassifier(estimators=[
#     ("cb", cb),
#     ('xgb', xgb),
#     ('rf', rf),
#     # ("svm", svm)
# ], voting='hard')
# ss = ShuffleSplit(train_size=0.8, test_size=0.1, n_splits=5)  # test:.1, train:.8, validation:.1
#
# score_xgb = cross_val_score(xgb, X, y, cv=ss)
# score_rf = cross_val_score(rf, X, y, cv=ss)
# # score_svm = cross_val_score(svm, X, y, cv=ss)
# score_cb = cross_val_score(cb, X, y, cv=ss)
# scores_hybrid = cross_val_score(voting_clf, X, y, cv=ss)
# print("Cross Validation Scores: ", scores_hybrid, score_xgb, score_rf, score_cb)
# print(
#     f"Average CV Score:\nHybrid: {scores_hybrid.mean()}\nXGB: {score_xgb.mean()}\nRF: {score_rf.mean()}\nCB: {score_cb.mean()}")
# print("Number of CV Scores used in Average: ", len(scores_hybrid))
# models = {"rf": rf, "xgb": xgb, "cb": cb, "hybrid": voting_clf}
#
# for model in models:
#     models[model].fit(X_train, y_train)
#     y_pred = models[model].predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"accuracy {model}: {accuracy}")
#     print(f"classification report {model}\n{classification_report(y_test, y_pred, zero_division=1)}")
# Train models and record accuracy

# Train and evaluate models across multiple training sessions
num_sessions = 100  # Number of training sessions
accuracy_results = {name: [] for name in ['RF', 'CatBoost', 'XGBoost', 'Voting']}

for i in range(num_sessions):
    xgb = XGBClassifier(**hyperparameters_XGB)
    rf = RandomForestClassifier(**hyperparameters_RFC)
    # svm = SVC(**hyperparameters_SVM)
    cb = CatBoostClassifier(**hyperparameters_CB)
    voting_clf = VotingClassifier(estimators=[
        ("cb", cb),
        ('xgb', xgb),
        ('rf', rf),
        # ("svm", svm)
    ], voting='hard')
    models = {'RF': rf, 'CatBoost': cb, 'XGBoost': xgb, 'Voting': voting_clf}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=i * 50)
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_results[name].append(accuracy)
# Convert accuracies to DataFrame and save to CSV
df_accuracies = pd.DataFrame(accuracy_results)
df_accuracies.to_csv('model_accuracies.csv', index=False)
# Convert accuracies to DataFrame and save to CSV
import matplotlib.pyplot as plt

# Read accuracies from CSV
df_accuracies = pd.read_csv('model_accuracies.csv')

# Plot
plt.figure(figsize=(10, 6),dpi = 72)
for column in df_accuracies.columns:
    plt.plot(df_accuracies.index, df_accuracies[column], label=column, marker='o')
plt.title('Model Accuracies Comparison')
plt.xlabel('Training Session')
plt.ylabel('Accuracy')
plt.xticks(df_accuracies.index)
plt.legend()
plt.grid(True)
plt.show()
