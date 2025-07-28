"""
Project Overview

The Project is based on the Stroke Prediction Dataset from Kaggle.
The Database includes medical, demographic, and behavioral information on approximately 5,110 patients, and records whether each patient has had a stroke (stroke = 1) or not.
The Database includes variables such as: age, sex, blood pressure, heart disease, blood glucose level, BMI, smoking status, marital status, residential area, and type of work.
In order to predict the likelihood of a stroke.

The task is to classify patients according to their risk of stroke using various machine learning algorithms, while analyzing the data, comparing the accuracy of the models, and testing additional techniques such as dimensionality reduction.

¥
Liron Cohen
Omer Apter
¥
"""

import random as rnd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import imblearn.pipeline as imbpipeline

gen = rnd.Random(69)
randomStates = np.array(
    [
        42,
        69,
        51,
        11,
        81,
        955,
        339,
        647,
        584,
        12,
        44,
        231,
        84,
        36,
        45,
        97,
        12,
        54,
        74,
        5421,
    ]
)  # Set a random states for reproducibility of experiments
state_random = 42

# Suppress all warnings
warnings.filterwarnings("ignore")

# Read from the dataset
dataset = pd.read_csv("healthcare-dataset-stroke-data.csv")


# ...existing code...
dataset.head()

# Display in a format with a given size and style
pd.set_option("display.float_format", lambda i: "%.2f" % i)
plt.rcParams["figure.figsize"] = 10, 3
sns.set_style("darkgrid")


# Function to get the n most important features from the dataset using ExtraTreesClassifier
def get_n_important_features(x, y, n=10):
    """Get the n most important features from the dataset using ExtraTreesClassifier."""
    ln = x.shape[1]
    model = ExtraTreesClassifier()
    model.fit(x, y)
    feat_importances = pd.Series(model.feature_importances_, index=x.columns)
    return feat_importances.nlargest(n).index.tolist()


# Function to run experiments with different models and return their scores
def run_experiments_gen(models: dict, metrics, iterations: 10, data_x, data_y):
    """Run experiments with different models and return their accuracies.
    Args:
        models (dict): Dictionary of model names and their instances model name:model.
        metrics (dict): Dictionary of metric names and their functions.
        iterations (int): Number of iterations to run the experiments.
        data_x: Features for training and testing.
        data_y: Labels for training and testing."""

    model_scores = {
        name: {metric: [] for metric in metrics.keys()} for name in models.keys()
    }

    print("Running experiments...")

    for iteration in range(iterations):
        # Splitting data into training and testing data
        X_train, X_test, Y_train, Y_test = train_test_split(
            data_x,
            data_y,
            test_size=0.2,
            stratify=data_y,
            random_state=randomStates[iteration],
        )

        # if variations["OverSampling"]:
        #     smote = SMOTE(random_state=randomStates[iteration], sampling_strategy=0.4)
        #     X_train, Y_train = smote.fit_resample(X_train, Y_train)
        # elif variations["OverUnderSampling"]:
        #     smoteenn = SMOTEENN(
        #         random_state=randomStates[iteration], sampling_strategy=0.6
        #     )
        #     X_train, Y_train = smoteenn.fit_resample(X_train, Y_train)

        print(f"iteration {iteration+1}/{iterations}")

        for name, model in models.items():
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)

            for metric_name, metric_func in metrics.items():
                score = metric_func(Y_test, Y_pred)
                model_scores[name][metric_name].append(score)
    # Calculate the average scores for each model
    for name, scores in model_scores.items():
        for metric_name, score_list in scores.items():
            model_scores[name][metric_name] = np.mean(score_list)
    # Return the accuracies of each model
    return model_scores


# grid search function to find the best parameters for each model
def runGridSearch(models: dict, parameters_grid, scoring_metrics, data_x, data_y):
    """Run GridSearchCV for each model and return the model.
    Args:
        models (dict): Dictionary of model names and their instances model name:model.
        parameters_grid (dict): Dictionary of model names and their parameter grids.
        scoring_metrics (dict): Dictionary of metric names and their functions.
        data_x: Features for training and testing.
        data_y: Labels for training and testing."""

    # scorers = {name: make_scorer(score) for name, score in scoring_metrics.items()}

    scorers = {
        "accuracy": make_scorer(accuracy_score, response_method="predict"),
        "precision": make_scorer(
            precision_score, response_method="predict", zero_division=0
        ),
        "recall": make_scorer(recall_score, response_method="predict", zero_division=0),
        "f1": make_scorer(f1_score, response_method="predict", zero_division=0),
    }

    results = {}
    for name, model in models.items():
        print(f"Running GridSearchCV for {name}...")
        grid_search = GridSearchCV(
            model,
            param_grid=parameters_grid[name],
            cv=5,
            scoring=scorers,
            refit="f1",
            n_jobs=-1,
        )
        grid_search.fit(data_x, data_y)
        results[name] = grid_search
        # print(f"Best params for {name}: {grid_search.best_params_}")
    return results


# pipe functions to apply transformations to the models
def add_PCA_transform(models: dict, n_components=14):
    """Apply PCA to the models and return the transformed models.
    Args:
        models (dict): Dictionary of model names and their instances model name:model.
    """
    pca_models = {}
    for name, model in models.items():
        pca_pipe = Pipeline(
            [
                ("StandardScaler", StandardScaler()),
                ("PCA", PCA(n_components=n_components, random_state=state_random)),
                ("clf", model),
            ]
        )
        pca_models[name] = pca_pipe
    return pca_models


def add_Smote_transform(models: dict, sampling_strategy=0.4):
    """Apply SMOTE oversampling to the models and return the transformed models.
    Args:
        models (dict): Dictionary of model names and their instances model name:model.
        sampling_strategy (float): The sampling strategy for SMOTE."""
    smote_models = {}
    for name, model in models.items():
        smote_pipe = imbpipeline.Pipeline(
            [
                ("StandardScaler", StandardScaler()),
                ("sample", SMOTE(sampling_strategy=sampling_strategy)),
                ("clf", model),
            ]
        )
        smote_models[name] = smote_pipe
    return smote_models


def add_SmoteENN_transform(models: dict, sampling_strategy=0.6):
    """Apply SMOTEENN overundersampling to the models and return the transformed models.
    Args:
        models (dict): Dictionary of model names and their instances model name:model.
        sampling_strategy (float): The sampling strategy for SMOTEENN."""
    smoteenn_models = {}
    for name, model in models.items():
        smoteenn_pipe = imbpipeline.Pipeline(
            [
                ("StandardScaler", StandardScaler()),
                ("sample", SMOTE(sampling_strategy=sampling_strategy)),
                ("clf", model),
            ]
        )
        smoteenn_models[name] = smoteenn_pipe
    return smoteenn_models


# end of functions defines

models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", solver="newton-cholesky", random_state=state_random
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=10,
        weights="distance",
        algorithm="auto",
        metric="minkowski",
    ),
    "Decision Tree": DecisionTreeClassifier(
        criterion="entropy",
        max_depth=100,
        class_weight="balanced",
        random_state=state_random,
    ),
    "Adaboost": AdaBoostClassifier(
        DecisionTreeClassifier(class_weight="balanced", max_depth=5),
        n_estimators=100,
        random_state=state_random,
        algorithm="SAMME",
    ),
    "SVM": LinearSVC(
        C=100,
        class_weight="balanced",
        random_state=state_random,
    ),
}

metrics_used = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
}

param_grids = {
    "Logistic Regression": {
        "solver": ["lbfgs", "liblinear", "newton-cholesky"],
        "C": [1, 10, 100, 1000],
    },
    "KNN": {
        "n_neighbors": [2, 3, 4, 5, 10, 15],
        "weights": ["uniform", "distance"],
        "metric": ["minkowski", "euclidean"],
    },
    "Decision Tree": {
        "criterion": ["gini", "entropy"],
        "max_depth": [10, 25, 50, 100, 150],
    },
    "Adaboost": {
        "n_estimators": [50, 75, 100, 150, 200, 300],
        "learning_rate": [0.5, 0.75, 1.0, 1.5, 2.0],
    },
    "SVM": {
        "C": [1, 10, 100, 1000],
        # "kernel": ["rbf", "linear"],
        # "degree": [2, 3],
    },
}

# REMOVE ID column -it's just an identifier and does not contribute to prediction(all other columns would shift left in index position)
dataset.drop("id", axis=1, inplace=True)
# Replace all text into numbers
dataset["bmi"] = pd.to_numeric(dataset["bmi"], errors="coerce")
# dataset['bmi'].fillna(0, inplace=True)
# Or use mean:
dataset["bmi"].fillna(dataset["bmi"].median(), inplace=True)
# dataset.dropna(inplace=True)
# Identify categorical columns
categorical_cols = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]

# preprocess the dataset

numeric_cols = ["age", "avg_glucose_level", "bmi"]
scaler = MinMaxScaler()

for col in numeric_cols:
    dataset[col] = scaler.fit_transform(dataset[[col]])

# One-hot encode categorical columns
dataset = pd.get_dummies(data=dataset, columns=categorical_cols)

# end of preprocessing

# Drop the 'stroke' column to use the rest as features
Y = dataset["stroke"]
X = dataset.drop(columns=["stroke"])

# Set the variations you want to test
variations = {
    "PCA": False,  # Set to True if you want to use PCA
    "PickBest": False,  # Set to True if you want to use the 10 best features
    "OverSampling": True,  # Set to True if you want to use SMOTE
    "OverUnderSampling": False,  # Set to True if you want to use SMOTEENN
    "UnderSampling": False,  # Set to True if you want to use RandomUnderSampler
}

if variations["PickBest"]:
    imp = [
        "avg_glucose_level",
        "bmi",
        "age",
        "hypertension",
        "heart_disease",
    ]  # manually selected features based on domain knowledge
    # Alternatively, you can use the get_n_important_features function to select the top features
    # imp = get_n_important_features(X, Y, 10)  # Get the 10 most important features
    X = dataset[imp]


if variations["PCA"]:
    models = add_PCA_transform(models, n_components=14)
    p_grid = {}  # Define the parameter grid for PCA
    for model_name in param_grids.keys():
        p_grid[model_name] = {}
        for param_name, params in param_grids[model_name].items():
            p_grid[model_name][f"clf__{param_name}"] = params
        p_grid[model_name]["PCA__n_components"] = [8, 9, 10, 11, 12, 13, 14]
    param_grids = p_grid  # Update the parameter grids with PCA parameters
    # pca_pipeline = make_pipeline(
    #     StandardScaler(), PCA(n_components=14, random_state=state_random)
    # )
    # X = pd.DataFrame(pca_pipeline.fit_transform(X))


if variations["OverSampling"]:
    models = add_Smote_transform(models, sampling_strategy=0.4)
    p_grid = {}
    for model_name in param_grids.keys():
        p_grid[model_name] = {}
        for param_name, params in param_grids.get(model_name, None).items():
            p_grid[model_name][f"clf__{param_name}"] = params
        p_grid[model_name]["sample__sampling_strategy"] = [0.2, 0.3, 0.4, 0.6, 0.8]
    param_grids = p_grid

elif variations["OverUnderSampling"]:
    models = add_SmoteENN_transform(models, sampling_strategy=0.6)
    p_grid = {}
    for model_name in param_grids.keys():
        p_grid[model_name] = {}
        for param_name, params in param_grids.get(model_name, None).items():
            p_grid[model_name][f"clf__{param_name}"] = params
        p_grid[model_name]["sample__sampling_strategy"] = [0.2, 0.3, 0.4, 0.6, 0.8]
    param_grids = p_grid

cv_results = runGridSearch(models, param_grids, metrics_used, X, Y)
# imporved_models = {}
for param_name, gridsearchcv in cv_results.items():
    print(f"Best params for {param_name}: {gridsearchcv.best_params_}")
    models.update(
        {param_name: gridsearchcv.best_estimator_}
    )  # Update the models with the best estimators

scores = run_experiments_gen(models, metrics_used, 20, X, Y)
scores_df = pd.DataFrame(scores).T  # Transpose to have models as rows
scores_df.index.name = "Model"

for model_name, value in variations.items():
    scores_df[model_name] = value  # Set the index name to 'Model'

# Save the results to a CSV file
scores_df.to_csv("Results.csv", mode="w", header=True, index=True)
print(scores_df.to_string())
# Make a list with the accuracies items
# acc_list = accuracies.items()
"""
k, v = zip(*acc_list)
v = np.round(a = v, decimals=2)  # Round the accuracies to 2 decimal places
# temp = pd.DataFrame(index=k, data=v, columns=["Accuracy"])
temp = pd.DataFrame(index=k, data=v)
# temp.sort_values(by=["Accuracy"], ascending=False, inplace=True)
"""
