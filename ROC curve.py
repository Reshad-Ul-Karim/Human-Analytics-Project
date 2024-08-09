import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, auc, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import seaborn as sns
import time
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier

# Load the dataset
df = pd.read_csv("COMBO2.csv")

# Split the dataset into features (X) and labels (y)


# Split the dataset into training and testing sets

hyperparameters_RFC = {'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2,
                       'n_estimators': 200,
                       }

hyperparameters_XGB = {'subsample': 0.7000000000000001, 'n_estimators': 200, 'max_depth': None, 'learning_rate': '0.05',
                       'gamma': 0.5, 'colsample_bytree': 0.6000000000000001, 'use_label_encoder': False,
                       'eval_metric': 'rmse', 'min_child_weight': 2, 'objective': 'binary:logistic',
                       }
hyperparameters_SVM = {'kernel': 'rbf', 'gamma': 1, 'C': 10}

# Initialize and train the RFC, SVM, and XGBoost models
models = [
    ("XGBoost", XGBClassifier(**hyperparameters_XGB)),
    ("Random Forest", RandomForestClassifier(**hyperparameters_RFC)),
    ("SVM", SVC(probability=True, random_state=42))
]
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
drop_columns = ['Filename', "is_stroke_face"]
c = 0
for region in ranked_regions:
    if c >= 3:
        for index in FACE_INDEXES[region]:
            drop_columns.append(f"{region}_{index}")
    c += 1
X = df.drop(drop_columns, axis=1)
y = df["is_stroke_face"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=150)
# X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=validation_ratio)

# for name, model in models:
#     start = time.time()
#     model.fit(X_train, y_train)
#     end = time.time()
#     print(f"Training time for {name}: {end - start:.2f} seconds")
#     y_pred = model.predict(X_test)
#     conf = confusion_matrix(y_test, y_pred)
#     labels = ["Non-Stroke", "Stroke", "Non-Stroke", "Stroke"]
#     group_percentages = ["{0:.2%}".format(value) for value in conf.flatten() / np.sum(conf)]
#     categories = ["Non_stroke", "Stroke"]
#     labels = np.asarray(labels).reshape(2, 2)
#     group_percentages = np.asarray(group_percentages).reshape(2, 2)
#     sns.heatmap(conf, annot=group_percentages, fmt="", cmap="Crimson", xticklabels=categories, yticklabels=categories)
#     plt.show()

# Plot ROC curve for each model
plt.figure(figsize=(10, 10))
#
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy for {name}: {accuracy * 100:.2f}%')
    print(classification_report(y_test, y_pred, zero_division=1))

    y_pred_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')



plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
