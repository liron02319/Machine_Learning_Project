'''
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
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC




state_random = 692  # Set a random state for reproducibility

# Suppress all warnings
warnings.filterwarnings("ignore")

# Read from the dataset
dataset = pd.read_csv('healthcare-dataset-stroke-data.csv')


# REMOVE ID column -it's just an identifier and does not contribute to prediction(all other columns would shift left in index position)
# dataset = pd.read_csv('healthcare-dataset-stroke-data.csv')
dataset.drop('id', axis=1, inplace=True)


# Replace all text into numbers
dataset['bmi'] = pd.to_numeric(dataset['bmi'], errors='coerce')
# dataset['bmi'].fillna(0, inplace=True)  
# Or use mean:
# dataset['bmi'].fillna(dataset['bmi'].median(), inplace=True)
dataset.dropna(inplace=True)


# ...existing code...

# Identify categorical columns
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# One-hot encode categorical columns
dataset = pd.get_dummies(data = dataset,dtype=int, columns=categorical_cols)

# ...existing code...
dataset.head()

# Display in a format with a given size and style
pd.set_option('display.float_format', lambda i: '%.2f' % i)
plt.rcParams['figure.figsize'] = 10, 3
sns.set_style("darkgrid")

def get_n_important_features(dataset_in:pd.DataFrame, n=10):
    """Get the n most important features from the dataset using ExtraTreesClassifier."""
    x = dataset_in.iloc[:, :-1]
    y = dataset_in.iloc[:, -1]
    ln = x.shape[1]
    model = ExtraTreesClassifier()
    model.fit(x, y)
    feat_importances = pd.Series(model.feature_importances_, index=x.columns)
    return feat_importances.nlargest(n).index.tolist() 

def plot_pca_explained_var(x):
    # Dimension reduction
    # Create scaler
    scaler = StandardScaler()
    # Create a PCA instance
    pca = PCA(n_components=10)
    # Create pipeline
    pipeline = make_pipeline(scaler, pca)
    # Fit the pipeline to samples
    pipeline.fit(x)
    # Plot the explained variances
    features = range(pca.n_components_)
    plt.bar(features, pca.explained_variance_)
    plt.xlabel('PCA feature')
    plt.ylabel('variance')
    plt.xticks(features)
    plt.show()

def make_pca(x):
    # Dimension reduction
    # Create scaler
    scaler = StandardScaler()
    # Create a PCA instance
    pca = PCA(n_components=10)
    # Create pipeline
    pipeline = make_pipeline(scaler, pca)
    # Fit the pipeline to samples
    return pipeline.fit_transform(x)

# Uncomment the next lines to plot the column importance, heart disease, and hypertension diagrams
# plot_column_importance(dataset)
# plot_heart_disease(dataset)
# plot_hypertension(dataset)

# ננקה את ערכי ה-None בעמודת bmi (כי היא לא רלוונטית פה)
# dataset['bmi'] = pd.to_numeric(dataset['bmi'], errors='coerce')
# dataset.dropna(subset=['bmi'], inplace=True)

# # נספור כמה בכל קבוצה
# group = dataset.groupby(['hypertension', 'stroke']).size().unstack(fill_value=0)

# # אחוז שבץ אצל כל קבוצה
# percent = (group[1] / (group[0] + group[1])) * 100
# print(percent)

###

def run_experiments_gen(models,metrics,iterations: 1, data_x, data_y):
    '''Run experiments with different models and return their accuracies.
    Args:
        models (dict): Dictionary of model names and their instances model name:model.
        iterations (int): Number of iterations to run the experiments.
        data_x: Features for training and testing.
        data_y: Labels for training and testing.'''
    
    # Splitting data into training and testing data
    X_train, X_test, Y_train, Y_test = train_test_split(data_x, data_y, test_size=0.2,stratify=data_y)
    model_scores = {name: {metric: [] for metric in metrics.keys()} for name in models.keys()}
    print("Running experiments...")
    for i in range(iterations):
        if i % 10 == 0:
            print(f"Iteration {i+1}/{iterations}")
        for name, model in models.items():
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)

            for metric_name, metric_func in metrics.items():
                score = metric_func(Y_test, Y_pred)
                model_scores[name][metric_name].append(score) # Multiply by 100 for percentage
    # Calculate the average scores for each model
    for name, scores in model_scores.items():
        for metric_name, score_list in scores.items():
            model_scores[name][metric_name] = np.mean(score_list)
    # Return the accuracies of each model
    return (model_scores)


def run_experiments(iterations: 1, data_x, data_y):

    # Splitting data into training and testing data
    X_train, X_test, Y_train, Y_test = train_test_split(data_x, data_y, test_size=0.2)

    accuracies = {"Logistic Regression": [],
                    "KNN": [],
                    "Decision Tree": [],
                    "Adaboost": [],
                    "SVM": []}
    
    print("Running experiments...")

    for i in range(iterations):
        if i % 10 == 0:
            print(f"Iteration {i+1}/{iterations}")
        # Logistic Regression
        model_Logistic = LogisticRegression()
        model_Logistic.fit(X_train, Y_train)
        Y_pred = model_Logistic.predict(X_test)
        model_Logistic_accuracy = round(accuracy_score(Y_test, Y_pred) * 100, 4)  # Accuracy
        accuracies['Logistic Regression'].append(model_Logistic_accuracy)

        # KNearest Neighbors
        model_KNN = KNeighborsClassifier(n_neighbors=15)
        model_KNN.fit(X_train, Y_train)
        Y_pred = model_KNN.predict(X_test)
        model_KNN_accuracy = round(accuracy_score(Y_test, Y_pred) * 100, 4)  # Accuracy
        accuracies['KNN'].append(model_KNN_accuracy)

        # Decision Tree
        model_tree = DecisionTreeClassifier(criterion="gini", max_depth=100)
        model_tree.fit(X_train, Y_train)
        Y_pred = model_tree.predict(X_test)
        model_tree_accuracy = round(accuracy_score(Y_test, Y_pred) * 100, 4)  # Accuracy
        accuracies['Decision Tree'].append(model_tree_accuracy)

        # Adaboost
        model_adaboost = AdaBoostClassifier(n_estimators=50, learning_rate=1)
        model_adaboost.fit(X_train, Y_train)
        Y_pred = model_adaboost.predict(X_test)
        Adaboost_accuracy = round(accuracy_score(Y_test, Y_pred) * 100, 4)  # Accuracy
        accuracies['Adaboost'].append(Adaboost_accuracy)

        # SVM
        model_svm = SVC(C=10000, kernel='rbf', degree=3)
        model_svm.fit(X_train, Y_train)
        Y_pred = model_svm.predict(X_test)
        svm_accuracy = round(accuracy_score(Y_test, Y_pred) * 100, 4)  # Accuracy
        accuracies['SVM'].append(svm_accuracy)
    return accuracies






models = {"Logistic Regression": LogisticRegression(class_weight='balanced',solver='newton-cholesky'),
                    "KNN": KNeighborsClassifier(n_neighbors=10, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', n_jobs=-1),
                    "Decision Tree": DecisionTreeClassifier(criterion="entropy", max_depth=100,class_weight='balanced'),
                    "Adaboost": AdaBoostClassifier(DecisionTreeClassifier(class_weight='balanced',max_depth=20),n_estimators=100, learning_rate=1),
                    "SVM": SVC(C=1, kernel='rbf', degree=3,class_weight='balanced')}

metrics_used = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score
}



imp = get_n_important_features(dataset, 10)  # Get the 10 most important features
# X = dataset[imp]
X = dataset.drop(columns=['stroke'])

X = pd.DataFrame(make_pca(X))

  # Drop the 'stroke' column to use the rest as features
Y = dataset['stroke']




scores = run_experiments_gen(models,metrics_used,100,X,Y)
scores_df = pd.DataFrame(scores).T  # Transpose to have models as rows
# df_acc = pd.DataFrame(accuracies)
# df_perc = pd.DataFrame(percisions)
# save_df = df_acc.join(df_perc, how='outer', lsuffix='_Accuracy', rsuffix='_Precision')
scores_df.to_csv('scores_10best_PCA.csv',mode='w',header=True ,index=True)
# Make a list with the accuracies items
# acc_list = accuracies.items()
'''
k, v = zip(*acc_list)
v = np.round(a = v, decimals=2)  # Round the accuracies to 2 decimal places
# temp = pd.DataFrame(index=k, data=v, columns=["Accuracy"])
temp = pd.DataFrame(index=k, data=v)
# temp.sort_values(by=["Accuracy"], ascending=False, inplace=True)
'''


def plot_accuracy_comparison(acc_df):
    # Plot accuracy for different models
    plt.figure(figsize=(18, 4))
    ACC = sns.barplot(y=acc_df.index, x=acc_df["Accuracy"].array, edgecolor="black", linewidth=3, orient="h",
                    palette="Set2")
    plt.ylabel("Model")
    plt.title("Algorithms Accuracy Comparison")
    plt.xlim(80, 100)

    ACC.spines['left'].set_linewidth(3)
    for w in ['right', 'top', 'bottom']:
        ACC.spines[w].set_visible(False)

    # Write text on barplots
    k = 0
    for ACC in ACC.patches:
        width = ACC.get_width()
        plt.text(width + 0.1, (ACC.get_y() + ACC.get_height() - 0.3), s="{}%".format(round(acc_df["Accuracy"][k],2)),
                fontname='monospace', fontsize=11, color='black')
        k += 1

    # plt.legend(loc="lower right")
    plt.tight_layout(pad=1.0)
    plt.show()

# plot_accuracy_comparison(temp)


# # Splitting data into training and testing data
# X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=state_random)

# # plot_pca_explained_var(X_train)

# # Logistic Regression
# model_Logistic = LogisticRegression(random_state=state_random)
# model_Logistic.fit(X_train, Y_train)
# Y_pred = model_Logistic.predict(X_test)
# model_Logistic_accuracy = round(accuracy_score(Y_test, Y_pred), 4) * 100  # Accuracy

# # KNearest Neighbors
# model_KNN = KNeighborsClassifier(n_neighbors=15)
# model_KNN.fit(X_train, Y_train)
# Y_pred = model_KNN.predict(X_test)
# model_KNN_accuracy = round(accuracy_score(Y_test, Y_pred), 4) * 100  # Accuracy

# # Decision Tree
# model_tree = DecisionTreeClassifier(random_state=state_random, criterion="gini", max_depth=100)
# model_tree.fit(X_train, Y_train)
# Y_pred = model_tree.predict(X_test)
# model_tree_accuracy = round(accuracy_score(Y_test, Y_pred), 4) * 100  # Accuracy

# # Adaboost
# model_adaboost = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=state_random)
# model_adaboost.fit(X_train, Y_train)
# Y_pred = model_adaboost.predict(X_test)
# Adaboost_accuracy = round(accuracy_score(Y_test, Y_pred), 4) * 100  # Accuracy

# # SVM
# model_svm = SVC(C=10000, kernel='rbf', degree=3)
# model_svm.fit(X_train, Y_train)
# Y_pred = model_svm.predict(X_test)
# svm_accuracy = round(accuracy_score(Y_test, Y_pred), 4) * 100  # Accuracy

# # Put into a dictionary the accuracies of each model
# accuracies = {"Logistic\n Regression": model_Logistic_accuracy,
#               "KNN": model_KNN_accuracy,
#               "Decision\nTree": model_tree_accuracy,
#               "Adaboost": Adaboost_accuracy,
#               "SVM": svm_accuracy}