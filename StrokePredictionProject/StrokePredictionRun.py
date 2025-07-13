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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC




# Suppress all warnings
warnings.filterwarnings("ignore")

# Read from the dataset
dataset = pd.read_csv('healthcare-dataset-stroke-data.csv')


# REMOVE ID column -it's just an identifier and does not contribute to prediction(all other columns would shift left in index position)
# dataset = pd.read_csv('healthcare-dataset-stroke-data.csv')
dataset.drop('id', axis=1, inplace=True)


# Replace all text into numbers

dataset['bmi'] = pd.to_numeric(dataset['bmi'], errors='coerce')
dataset['bmi'].fillna(0, inplace=True)  # Or use mean: dataset['bmi'].fillna(dataset['bmi'].mean(), inplace=True)



dataset['gender'].replace(['Female'], 1, inplace=True)
dataset['gender'].replace(['Male'], 2, inplace=True)
dataset['gender'].replace(['Other'], 3, inplace=True)

dataset['ever_married'].replace(['Yes'], 4, inplace=True)
dataset['ever_married'].replace(['No'], 5, inplace=True)

dataset['work_type'].replace(['children'], 6, inplace=True)
dataset['work_type'].replace(['Govt_job'], 7, inplace=True)
dataset['work_type'].replace(['Never_worked'], 8, inplace=True)
dataset['work_type'].replace(['Private'], 9, inplace=True)
dataset['work_type'].replace(['Self-employed'], 10, inplace=True)

dataset['Residence_type'].replace(['Rural'], 11, inplace=True)
dataset['Residence_type'].replace(['Urban'], 12, inplace=True)

dataset['smoking_status'].replace(['formerly smoked'], 13, inplace=True)
dataset['smoking_status'].replace(['never smoked'], 14, inplace=True)
dataset['smoking_status'].replace(['smokes'], 15, inplace=True)
dataset['smoking_status'].replace(['Unknown'], 16, inplace=True)
dataset.head()

# Display in a format with a given size and style
pd.set_option('display.float_format', lambda i: '%.2f' % i)
plt.rcParams['figure.figsize'] = 10, 3
sns.set_style("darkgrid")

# Show width
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Checking the importance of each column with ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(x, y)
# Print the importance of each column by its order
print(model.feature_importances_/model.feature_importances_.sum())
# Do a pie model of size 10 of the most to the least importance from the columns dataset
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(10).plot(kind='pie')
plt.show()
######## Diagram Heart Disease

# קיבוץ לפי מחלת לב ושבץ מוחי
grouped = dataset.groupby(['heart_disease', 'stroke']).size().unstack(fill_value=0)

# שינוי שמות לעברית
grouped.index = ['No Heart Disease', 'Yes Heart Disease']
grouped.columns = ['No', 'Yes']

# ציור הגרף
ax = grouped.plot(kind='bar', figsize=(8, 6), color=['skyblue', 'salmon'])

# הוספת כותרות
plt.title('Status: Heart Disease Of The Patients:')
plt.ylabel('Number Of Patients')
plt.xticks(rotation=0)
plt.legend(title='Status - Stroke')
plt.grid(axis='y')
plt.tight_layout()

# הוספת המספרים מעל כל עמודה
for container in ax.containers:
    for bar in container:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # מרווח אנכי
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

plt.show()

# ננקה את ערכי ה-None בעמודת bmi (כי היא לא רלוונטית פה)
dataset['bmi'] = pd.to_numeric(dataset['bmi'], errors='coerce')
dataset.dropna(subset=['bmi'], inplace=True)

# נספור כמה בכל קבוצה
group = dataset.groupby(['heart_disease', 'stroke']).size().unstack(fill_value=0)

# אחוז שבץ אצל כל קבוצה
percent = (group[1] / (group[0] + group[1])) * 100
print(percent)


######## Diagram Hypertention

# קיבוץ לפי לחץ דם גבוה ושבץ מוחי
grouped = dataset.groupby(['hypertension', 'stroke']).size().unstack(fill_value=0)

# שינוי שמות לעברית
grouped.index = ['No Hypertension', 'Yes Hypertension']
grouped.columns = ['No', 'Yes']

# ציור הגרף
ax = grouped.plot(kind='bar', figsize=(8, 6), color=['skyblue', 'salmon'])

# הוספת כותרות
plt.title('Status: Hypertension Of The Patients:')
plt.ylabel('Number Of Patients')
plt.xticks(rotation=0)
plt.legend(title='Status - Stroke')
plt.grid(axis='y')
plt.tight_layout()

# הוספת המספרים מעל כל עמודה
for container in ax.containers:
    for bar in container:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

plt.show()

# ננקה את ערכי ה-None בעמודת bmi (כי היא לא רלוונטית פה)
dataset['bmi'] = pd.to_numeric(dataset['bmi'], errors='coerce')
dataset.dropna(subset=['bmi'], inplace=True)

# נספור כמה בכל קבוצה
group = dataset.groupby(['hypertension', 'stroke']).size().unstack(fill_value=0)

# אחוז שבץ אצל כל קבוצה
percent = (group[1] / (group[0] + group[1])) * 100
print(percent)

###


# Using the important columns form the pie shown and reference it to the stroke column
x = dataset[['avg_glucose_level', 'smoking_status', 'work_type', 'bmi', 'age']]
## # Uncomment the next line to use PCA for dimensionality reduction instead of the original features
# x = PCA(n_components=3).fit_transform(x)
y = dataset['stroke']

# Splitting data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=65)

# Dimension reduction
# Create scaler
scaler = StandardScaler()
# Create a PCA instance
pca = PCA()
# Create pipeline
pipeline = make_pipeline(scaler, pca)
# Fit the pipeline to samples
pipeline.fit(X_train)
# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()

# Logistic Regression
model_Logistic = LogisticRegression(random_state=1)
model_Logistic.fit(X_train, Y_train)
Y_pred = model_Logistic.predict(X_test)
model_Logistic_accuracy = round(accuracy_score(Y_test, Y_pred), 4) * 100  # Accuracy

# KNearest Neighbors
model_KNN = KNeighborsClassifier(n_neighbors=15)
model_KNN.fit(X_train, Y_train)
Y_pred = model_KNN.predict(X_test)
model_KNN_accuracy = round(accuracy_score(Y_test, Y_pred), 4) * 100  # Accuracy

# Decision Tree
model_tree = DecisionTreeClassifier(random_state=10, criterion="gini", max_depth=100)
model_tree.fit(X_train, Y_train)
Y_pred = model_tree.predict(X_test)
model_tree_accuracy = round(accuracy_score(Y_test, Y_pred), 4) * 100  # Accuracy

# Adaboost
model_adaboost = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0)
model_adaboost.fit(X_train, Y_train)
Y_pred = model_adaboost.predict(X_test)
Adaboost_accuracy = round(accuracy_score(Y_test, Y_pred), 4) * 100  # Accuracy

# SVM
model_svm = SVC(C=10000, kernel='rbf', degree=3)
model_svm.fit(X_train, Y_train)
Y_pred = model_svm.predict(X_test)
svm_accuracy = round(accuracy_score(Y_test, Y_pred), 4) * 100  # Accuracy

# Put into a dictionary the accuracies of each model
accuracies = {"Logistic\n Regression": model_Logistic_accuracy,
              "KNN": model_KNN_accuracy,
              "Decision\nTree": model_tree_accuracy,
              "Adaboost": Adaboost_accuracy,
              "SVM": svm_accuracy}

# Make a list with the accuracies items
acc_list = accuracies.items()
k, v = zip(*acc_list)
temp = pd.DataFrame(index=k, data=v, columns=["Accuracy"])
temp.sort_values(by=["Accuracy"], ascending=False, inplace=True)

# Plot accuracy for different models
plt.figure(figsize=(18, 4))
ACC = sns.barplot(y=temp.index, x=temp["Accuracy"].array, edgecolor="black", linewidth=3, orient="h",
                  palette="Set2")
plt.ylabel("Model")
plt.title("Algorithms Accuracy Comparison")
plt.xlim(80, 98)

ACC.spines['left'].set_linewidth(3)
for w in ['right', 'top', 'bottom']:
    ACC.spines[w].set_visible(False)

# Write text on barplots
k = 0
for ACC in ACC.patches:
    width = ACC.get_width()
    plt.text(width + 0.1, (ACC.get_y() + ACC.get_height() - 0.3), s="{}%".format(round(temp["Accuracy"][k],2)),
             fontname='monospace', fontsize=11, color='black')
    k += 1

# plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
