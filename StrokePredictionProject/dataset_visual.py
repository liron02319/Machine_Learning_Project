import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

# Suppress all warnings
warnings.filterwarnings("ignore")

# Read from the dataset
dataset = pd.read_csv('healthcare-dataset-stroke-data.csv')


# REMOVE ID column -it's just an identifier and does not contribute to prediction(all other columns would shift left in index position)
# dataset = pd.read_csv('healthcare-dataset-stroke-data.csv')
dataset.drop('id', axis=1, inplace=True)

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

def plot_column_importance(dataset_in):
    # Checking the importance of each column with ExtraTreesClassifier
    x = dataset_in.iloc[:, :-1]
    y = dataset_in.iloc[:, -1]
    # Checking the importance of each column with ExtraTreesClassifier
    model = ExtraTreesClassifier()
    model.fit(x, y)
    # Print the importance of each column by its order
    sum = model.feature_importances_.sum()
    print(model.feature_importances_)
    # Do a pie model of size 10 of the most to the least importance from the columns dataset
    feat_importances = pd.Series(model.feature_importances_, index=x.columns)
    feat_importances.plot(kind='pie')
    plt.show()


######## Diagram Heart Disease
def plot_heart_disease(dataset_in):
    # קיבוץ לפי מחלת לב ושבץ מוחי
    grouped = dataset_in.groupby(['heart_disease', 'stroke']).size().unstack(fill_value=0)

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
    dataset_in['bmi'] = pd.to_numeric(dataset_in['bmi'], errors='coerce')
    dataset_in.dropna(subset=['bmi'], inplace=True)

    # נספור כמה בכל קבוצה
    group = dataset_in.groupby(['heart_disease', 'stroke']).size().unstack(fill_value=0)

    # אחוז שבץ אצל כל קבוצה
    percent = (group[1] / (group[0] + group[1])) * 100
    print(percent)


######## Diagram Hypertention
def plot_hypertension(dataset_in):
    # קיבוץ לפי לחץ דם גבוה ושבץ מוחי
    grouped = dataset_in.groupby(['hypertension', 'stroke']).size().unstack(fill_value=0)

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


sns.set_theme(style="ticks")

# df = sns.load_dataset("penguins")
# # plt.figure(figsize=(12, 6))

plot_heart_disease(dataset)
