import pandas as pd

# Load the datasets
df_straight = pd.read_csv('flattened_straight face_landmarks.csv')
df_droopy = pd.read_csv('flattened_droopy_landmarks.csv')

# Add a label column, 0 for straight faces and 1 for droopy faces
df_straight['label'] = 0
df_droopy['label'] = 1

# Combine the two datasets
df = pd.concat([df_straight, df_droopy], ignore_index=True)

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

"""splitting the dataset"""

from sklearn.model_selection import train_test_split

# Exclude the 'Filename' column from the features
X = df.drop(['label', 'Filename'], axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""training the model using XGBoost"""
import xgboost as xgb

# Define the model
#model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, enable_categorical=True)

# Train the model
model.fit(X_train, y_train)

"evaluating the model"

from sklearn.metrics import accuracy_score, classification_report

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, zero_division=1))
