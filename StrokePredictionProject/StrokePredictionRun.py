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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

gen = rnd.Random(69)
randomStates = np.array(
    [42, 69, 51, 11, 81, 955, 339, 647, 584, 12,44,231,84,36,45,97,12,54,74,5421]
)  # Set a random states for reproducibility of experiments
rndstate_of_model = 42

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


def get_n_important_features(x, y, n=10):
    """Get the n most important features from the dataset using ExtraTreesClassifier."""
    ln = x.shape[1]
    model = ExtraTreesClassifier()
    model.fit(x, y)
    feat_importances = pd.Series(model.feature_importances_, index=x.columns)
    return feat_importances.nlargest(n).index.tolist()


def run_experiments_gen(
    models:dict, metrics, iterations: 10, data_x, data_y
):
    """Run experiments with different models and return their accuracies.
    Args:
        models (dict): Dictionary of model names and their instances model name:model.
        metrics (dict): Dictionary of metric names and their functions.
        split_iterations (int): Number of iterations to split the data.
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

        if variations["OverSampling"]:
            smote = SMOTE(random_state=randomStates[iteration])
            X_train, Y_train = smote.fit_resample(X_train, Y_train)
        elif variations["OverUnderSampling"]:
            smoteenn = SMOTEENN(random_state=randomStates[iteration])
            X_train, Y_train = smoteenn.fit_resample(X_train, Y_train)

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


def run_experiments(iterations: 1, data_x, data_y):

    # Splitting data into training and testing data
    X_train, X_test, Y_train, Y_test = train_test_split(data_x, data_y, test_size=0.2)

    accuracies = {
        "Logistic Regression": [],
        "KNN": [],
        "Decision Tree": [],
        "Adaboost": [],
        "SVM": [],
    }

    print("Running experiments...")

    for i in range(iterations):
        if i % 10 == 0:
            print(f"Iteration {i+1}/{iterations}")
        # Logistic Regression
        model_Logistic = LogisticRegression()
        model_Logistic.fit(X_train, Y_train)
        Y_pred = model_Logistic.predict(X_test)
        model_Logistic_accuracy = round(
            accuracy_score(Y_test, Y_pred) * 100, 4
        )  # Accuracy
        accuracies["Logistic Regression"].append(model_Logistic_accuracy)

        # KNearest Neighbors
        model_KNN = KNeighborsClassifier(n_neighbors=15)
        model_KNN.fit(X_train, Y_train)
        Y_pred = model_KNN.predict(X_test)
        model_KNN_accuracy = round(accuracy_score(Y_test, Y_pred) * 100, 4)  # Accuracy
        accuracies["KNN"].append(model_KNN_accuracy)

        # Decision Tree
        model_tree = DecisionTreeClassifier(criterion="gini", max_depth=100)
        model_tree.fit(X_train, Y_train)
        Y_pred = model_tree.predict(X_test)
        model_tree_accuracy = round(accuracy_score(Y_test, Y_pred) * 100, 4)  # Accuracy
        accuracies["Decision Tree"].append(model_tree_accuracy)

        # Adaboost
        model_adaboost = AdaBoostClassifier(n_estimators=50, learning_rate=1)
        model_adaboost.fit(X_train, Y_train)
        Y_pred = model_adaboost.predict(X_test)
        Adaboost_accuracy = round(accuracy_score(Y_test, Y_pred) * 100, 4)  # Accuracy
        accuracies["Adaboost"].append(Adaboost_accuracy)

        # SVM
        model_svm = SVC(C=10000, kernel="rbf", degree=3)
        model_svm.fit(X_train, Y_train)
        Y_pred = model_svm.predict(X_test)
        svm_accuracy = round(accuracy_score(Y_test, Y_pred) * 100, 4)  # Accuracy
        accuracies["SVM"].append(svm_accuracy)
    return accuracies


models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", solver="newton-cholesky",random_state=rndstate_of_model
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=10,
        weights="distance",
        algorithm="auto",
        metric="minkowski",
    ),
    "Decision Tree": DecisionTreeClassifier(
        criterion="entropy", max_depth=100, class_weight="balanced",random_state=rndstate_of_model
    ),
    "Adaboost": AdaBoostClassifier(
        DecisionTreeClassifier(class_weight="balanced", max_depth=5),
        n_estimators=100,
        random_state=rndstate_of_model,
    ),
    "SVM": SVC(C=100, kernel="rbf", degree=3, class_weight="balanced",random_state=rndstate_of_model),
}


#end of functions defines

metrics_used = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
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

numeric_cols = ["age", "avg_glucose_level", "bmi"]
scaler = MinMaxScaler()

for col in numeric_cols:
    dataset[col] = scaler.fit_transform(dataset[[col]])

# One-hot encode categorical columns
dataset = pd.get_dummies(data=dataset, columns=categorical_cols)

# Drop the 'stroke' column to use the rest as features
Y = dataset["stroke"]
X = dataset.drop(columns=["stroke"])

variations = {
    "PCA": False ,  # Set to True if you want to use PCA
    "PickBest": False,  # Set to True if you want to use the 10 best features
    "OverSampling": True,  # Set to True if you want to use SMOTE
    "OverUnderSampling": False,  # Set to True if you want to use SMOTEENN
    "UnderSampling": False,  # Set to True if you want to use RandomUnderSampler
}

if variations["PickBest"]:
    imp = ["avg_glucose_level", "bmi", "age", "hypertension", "heart_disease"]
    # imp = get_n_important_features(X, Y, 10)  # Get the 10 most important features
    X = dataset[imp]


if variations["PCA"]:
    pca_pipeline = make_pipeline(StandardScaler(), PCA(n_components=14))
    X = pd.DataFrame(pca_pipeline.fit_transform(X))

scores = run_experiments_gen(models, metrics_used, 20, X, Y)
scores_df = pd.DataFrame(scores).T  # Transpose to have models as rows
scores_df.index.name = "Model"

for key, value in variations.items():
    scores_df[key] = value  # Set the index name to 'Model'

scores_df.to_csv("Results.csv", mode="w", header=True, index=True)
# Make a list with the accuracies items
# acc_list = accuracies.items()
"""
k, v = zip(*acc_list)
v = np.round(a = v, decimals=2)  # Round the accuracies to 2 decimal places
# temp = pd.DataFrame(index=k, data=v, columns=["Accuracy"])
temp = pd.DataFrame(index=k, data=v)
# temp.sort_values(by=["Accuracy"], ascending=False, inplace=True)
"""
