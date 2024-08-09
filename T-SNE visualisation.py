import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
# Load data
df = pd.read_csv('COMBO2.csv')
drop_columns = ['Filename', "is_stroke_face"]
X = df.drop(drop_columns, axis=1)
y = df["is_stroke_face"]

# Hyperparameters
hyperparameters_RFC = {
    'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2,
    'n_estimators': 200
}
hyperparameters_XGB = {
    'subsample': 0.7, 'n_estimators': 200, 'max_depth': None, 'learning_rate': 0.05,
    'gamma': 0.5, 'colsample_bytree': 0.6, 'use_label_encoder': False,
    'eval_metric': 'rmse', 'min_child_weight': 2, 'objective': 'binary:logistic'
}
hyperparameters_CB = {
    'bagging_temperature': 0.8607305832563434, 'bootstrap_type': 'MVS',
    'colsample_bylevel': 0.917411003148779, 'depth': 8, 'grow_policy': 'SymmetricTree',
    'iterations': 918, 'l2_leaf_reg': 8, 'learning_rate': 0.29287291117375575, 'max_bin': 231,
    'min_data_in_leaf': 9, 'od_type': 'Iter', 'od_wait': 21, 'one_hot_max_size': 7,
    'random_strength': 0.6963042728397884, 'scale_pos_weight': 1.924541179848884, 'subsample': 0.6480869299533999
}

# Define models
models = {
    'RandomForest': RandomForestClassifier(**hyperparameters_RFC),
    'XGBoost': XGBClassifier(**hyperparameters_XGB),
    'CatBoost': CatBoostClassifier(**hyperparameters_CB),
    'VotingClassifier': VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(**hyperparameters_RFC)),
        ('xgb', XGBClassifier(**hyperparameters_XGB)),
        ('cb', CatBoostClassifier(**hyperparameters_CB))
    ], voting='soft')
}

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=150)

# Training and collecting feature representations
features_epochs = {model_name: [] for model_name in models}
epochs_to_visualize = [50, 100, 150, 200]  # Specific epochs to visualize

# Simulate training at specific epochs
# Assuming the rest of your code is the same as before and correctly implemented

# Simulate training at specific epochs
for epoch in epochs_to_visualize:
    for name, model in models.items():
        model.fit(X_train, y_train)  # Train model
        if hasattr(model, 'decision_function'):
            features = model.decision_function(X_test)
            # Reshape decision function output to be two-dimensional if necessary
            if features.ndim == 1:
                features = features.reshape(-1, 1)
        else:
            # Use both output probabilities for binary classification
            features = model.predict_proba(X_test)
            if features.shape[1] == 2:
                # Use both columns for t-SNE
                pass  # No need to reshape, as it's already two-dimensional
            else:
                # For multi-class, handle appropriately, possibly using a different method
                features = np.max(features, axis=1).reshape(-1, 1)  # Example adjustment

        features_epochs[name].append(features)

        y_pred = model.predict(X_test)
        print(f"Epoch {epoch}, {name} Accuracy: {accuracy_score(y_test, y_pred)}")

# Applying T-SNE and plotting
for name, epochs_features in features_epochs.items():
    plt.figure(figsize=(15, 10))
    for i, features in enumerate(epochs_features):
        tsne = TSNE(n_components=2, random_state=42)
        X_embedded = tsne.fit_transform(features)  # Directly use features without reshaping
        plt.subplot(2, 5, i + 1)
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_test, cmap='viridis')
        plt.title(f'{name} Epoch {epochs_to_visualize[i]}')
    plt.show()

