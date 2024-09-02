import pandas as pd
from sklearn.model_selection import train_test_split
import os
import hashlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import logging

# Load and prepare data
df = pd.read_csv("COMBO2.csv")
drop_columns = ['Filename', "is_stroke_face"]
X = df.drop(drop_columns, axis=1)
y = df["is_stroke_face"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=150)

def train_rf(config):
    try:
        rf = RandomForestClassifier(
            n_estimators=int(config["n_estimators"]),
            max_depth=int(config["max_depth"]) if config["max_depth"] is not None else None,
            min_samples_split=int(config["min_samples_split"]),
            min_samples_leaf=int(config["min_samples_leaf"]),
            max_features=config["max_features"],
            bootstrap=config["bootstrap"],
            criterion=config["criterion"],
            random_state=150
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return {"accuracy": accuracy}
    except Exception as e:
        logging.error(f"Error in train_rf: {str(e)}")
        raise e

# Define expanded search space for RandomForest
search_space = {
    "n_estimators": tune.choice([100, 200, 300, 400, 500, 600, 700, 800]),
    "max_depth": tune.choice([None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
    "min_samples_split": tune.choice([2, 4, 6, 8, 10, 12, 14, 16]),
    "min_samples_leaf": tune.choice([1, 2, 3, 4, 5, 6, 7, 8]),
    "max_features": tune.choice(['auto', 'sqrt', 'log2', 0.25, 0.5, 0.75]),
    "bootstrap": tune.choice([True, False]),
    "criterion": tune.choice(['gini', 'entropy'])
}

# Initialize Ray
ray.init(logging_level=logging.INFO)

# Run hyperparameter tuning
try:
    analysis = tune.run(
        train_rf,
        config=search_space,
        num_samples=200,
        scheduler=ASHAScheduler(metric="accuracy", mode="max", max_t=1000),
        search_alg=OptunaSearch(metric="accuracy", mode="max"),
        resources_per_trial={"cpu": 4, "gpu": 0},
        verbose=2,
        trial_dirname_creator=lambda trial: hashlib.md5(trial.trial_id.encode()).hexdigest()[:10],
        storage_path=os.path.expanduser("~/ray_results"),
        raise_on_failed_trial=False
    )
    print("Ray Tune completed.")

    best_config = analysis.get_best_config(metric="accuracy", mode="max")
    print("Best hyperparameters found:", best_config)

    # Train the final model with the best hyperparameters
    best_rf = RandomForestClassifier(**best_config, random_state=150)
    best_rf.fit(X_train, y_train)
    final_accuracy = accuracy_score(y_test, best_rf.predict(X_test))
    print(f"Final model accuracy: {final_accuracy}")

except Exception as e:
    print(f"An error occurred during Ray Tune execution: {str(e)}")

finally:
    ray.shutdown()