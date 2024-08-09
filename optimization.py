import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import numpy as np

df = pd.read_csv('COMBO.csv')
X = df.drop(['Filename', "is_stroke_face"], axis=1)

y = df["is_stroke_face"]
rf = RandomForestClassifier()

# Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid = GridSearchCV(rf, param_grid=param_grid, cv=3, scoring='accuracy')
model_grid = grid.fit(X, y)
print('Best hyperparameters are: ' + str(model_grid.best_params_))
print('Best score is: ' + str(model_grid.best_score_))
'''
# Random Search
rs_space = {'max_depth': list(np.arange(10, 100, step=10)) + [None],
            'n_estimators': np.arange(10, 500, step=50),
            'max_features': randint(1, 7),
            'criterion': ['gini', 'entropy'],
            'min_samples_leaf': randint(1, 4),
            'min_samples_split': np.arange(2, 10, step=2)
            }
rf_random = RandomizedSearchCV(rf, rs_space, n_iter=500, scoring='accuracy', n_jobs=-1, cv=3)
model_random = rf_random.fit(X, y)
print('Best hyperparameters are: ' + str(model_random.best_params_))
print('Best score is: ' + str(model_random.best_score_))
'''