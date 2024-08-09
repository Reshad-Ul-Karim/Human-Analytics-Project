import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import randint, uniform

# Load data
df = pd.read_csv('COMBO2.csv')

# Define features to be dropped
drop_columns = ['Filename', 'is_stroke_face']

# Dynamically build the drop_columns list based on FACE_INDEXES and ranked_regions
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
# Load and prepare the data
drop_columns = ['Filename', "is_stroke_face"]
df = pd.read_csv('COMBO2.csv')
# c = 0
# for region in ranked_regions:
#     if c >= 3:
#         for index in FACE_INDEXES[region]:
#             drop_columns.append(f"{region}_{index}")
#     c += 1
X = df.drop(drop_columns, axis=1)
y = df["is_stroke_face"]
X = df.drop(drop_columns, axis=1)
y = df["is_stroke_face"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=150)

# Define the parameter distribution
param_dist = {
    'iterations': randint(100, 500),
    'depth': randint(4, 10),
    'learning_rate': uniform(0.01, 0.3),
    'l2_leaf_reg': randint(1, 10),
    'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS'],
    'subsample': uniform(0.6, 0.4),  # Ensuring subsample is always between 0.6 and 1.0
    'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
    'od_type': ['IncToDec', 'Iter'],
    'od_wait': randint(10, 50)
}

# Initialize CatBoostClassifier
model = CatBoostClassifier(verbose=False, thread_count=-1)

# Randomized Search CV
random_search_cb = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,  # Reduced number of iterations for brevity
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    error_score='raise'  # To trace errors more effectively
)

# Fit RandomizedSearchCV to the training data
random_search_cb.fit(X_train, y_train)

# Evaluate the best model
best_model = random_search_cb.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Output the results
print("Best Parameters:", random_search_cb.best_params_)
print("Best Cross-validation Score: {:.2f}".format(random_search_cb.best_score_))
print("Test Accuracy: {:.2f}".format(accuracy))
print("Classification Report:\n", classification_report(y_test, y_pred))
