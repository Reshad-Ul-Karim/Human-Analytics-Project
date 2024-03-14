import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV

"""Step 1: Data Preparation"""

df_straight = pd.read_csv('flattened_straight face_landmarks.csv')
df_droopy = pd.read_csv('flattened_droopy_landmarks.csv')

df_straight['label'] = 0
df_droopy['label'] = 1

df = pd.concat([df_straight, df_droopy], ignore_index=True)
print(df)
df = df.sample(frac=1).reset_index(drop=True)



#Step 2: Splitting the Dataset

# Exclude the 'Filename' column from the features
X = df.drop(['label', 'Filename'], axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Step 3: Hyperparameter Tuning Setup

param_dist = {
    'n_estimators': randint(100, 1000),
    'learning_rate': uniform(0.01, 0.3),
    'max_depth': randint(3, 10),
    'min_child_weight': randint(1, 6),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': uniform(0, 0.5)
}

#model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, enable_categorical=True)

n_iter_search = 20
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=n_iter_search,
    scoring='accuracy',
    cv=5,
    random_state=42,
    verbose=3
)

#Step 4: Perform Hyperparameter Tuning

random_search.fit(X_train, y_train)

#Step 5: Training the Best Model

best_model = random_search.best_estimator_
best_model.fit(X_train, y_train)  # Optional, as the best estimator is already fitted

#Step 6: Evaluating the Model

y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Model Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, zero_division=1))

#Optional: Review Best Parameters


#print("Best Parameters:", random_search.best_params_)
#print("Best Estimator:", random_search.best_estimator_)

