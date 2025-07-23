import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline

# Suppress all warnings
warnings.filterwarnings("ignore")

# Read from the dataset
dataset = pd.read_csv("healthcare-dataset-stroke-data.csv")


# REMOVE ID column -it's just an identifier and does not contribute to prediction(all other columns would shift left in index position)
# dataset = pd.read_csv('healthcare-dataset-stroke-data.csv')
dataset.drop("id", axis=1, inplace=True)

dataset["bmi"] = pd.to_numeric(dataset["bmi"], errors="coerce")
# dataset["bmi"].fillna(0, inplace=True)  
# Or use mean: 
dataset['bmi'].fillna(dataset['bmi'].mean(), inplace=True)


dataset["gender"].replace(["Female"], 1, inplace=True)
dataset["gender"].replace(["Male"], 2, inplace=True)
dataset["gender"].replace(["Other"], 3, inplace=True)

dataset["ever_married"].replace(["Yes"], 4, inplace=True)
dataset["ever_married"].replace(["No"], 5, inplace=True)

dataset["work_type"].replace(["children"], 6, inplace=True)
dataset["work_type"].replace(["Govt_job"], 7, inplace=True)
dataset["work_type"].replace(["Never_worked"], 8, inplace=True)
dataset["work_type"].replace(["Private"], 9, inplace=True)
dataset["work_type"].replace(["Self-employed"], 10, inplace=True)

dataset["Residence_type"].replace(["Rural"], 11, inplace=True)
dataset["Residence_type"].replace(["Urban"], 12, inplace=True)

dataset["smoking_status"].replace(["formerly smoked"], 13, inplace=True)
dataset["smoking_status"].replace(["never smoked"], 14, inplace=True)
dataset["smoking_status"].replace(["smokes"], 15, inplace=True)
dataset["smoking_status"].replace(["Unknown"], 16, inplace=True)


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
    feat_importances.plot(kind="pie")
    plt.show()


### Plotting the distribution of stroke cases in the dataset
def plot_stroke_distribution(dataset_in):
    """Plot the distribution of stroke cases in the dataset."""
    stroke_counts = dataset_in["stroke"].value_counts()
    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=stroke_counts.index, y=stroke_counts.values, palette=["skyblue", "salmon"]
    )
    plt.xticks([0, 1], ["No Stroke", "Stroke"])
    plt.title("Distribution of Stroke Cases")
    plt.ylabel("Number of Patients")
    plt.xlabel("Stroke Status")
    for i, v in enumerate(stroke_counts.values):
        plt.text(i, v + 10, str(v), ha="center", fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_pca_explained_var(x):
    # Dimension reduction
    # Create scaler
    scaler = StandardScaler()
    # Create a PCA instance
    pca = PCA(n_components=5)
    # Create pipeline
    pipeline = make_pipeline(scaler, pca)
    # Fit the pipeline to samples
    pipeline.fit(x)
    # Plot the explained variances
    features = range(pca.n_components_)
    plt.bar(features, pca.explained_variance_)
    plt.xlabel("PCA feature")
    plt.ylabel("variance")
    plt.xticks(features)
    plt.show()


######## Diagram Heart Disease
def plot_heart_disease(dataset_in):
    # קיבוץ לפי מחלת לב ושבץ מוחי
    grouped = (
        dataset_in.groupby(["heart_disease", "stroke"]).size().unstack(fill_value=0)
    )

    # שינוי שמות לעברית
    grouped.index = ["No Heart Disease", "Yes Heart Disease"]
    grouped.columns = ["No", "Yes"]

    # ציור הגרף
    ax = grouped.plot(kind="bar", figsize=(8, 6), color=["skyblue", "salmon"])

    # הוספת כותרות
    plt.title("Status: Heart Disease Of The Patients:")
    plt.ylabel("Number Of Patients")
    plt.xticks(rotation=0)
    plt.legend(title="Status - Stroke")
    plt.grid(axis="y")
    plt.tight_layout()

    # הוספת המספרים מעל כל עמודה
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            ax.annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # מרווח אנכי
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.show()

    # ננקה את ערכי ה-None בעמודת bmi (כי היא לא רלוונטית פה)
    dataset_in["bmi"] = pd.to_numeric(dataset_in["bmi"], errors="coerce")
    dataset_in.dropna(subset=["bmi"], inplace=True)

    # נספור כמה בכל קבוצה
    group = dataset_in.groupby(["heart_disease", "stroke"]).size().unstack(fill_value=0)

    # אחוז שבץ אצל כל קבוצה
    percent = (group[1] / (group[0] + group[1])) * 100
    print(percent)


######## Diagram Hypertention
def plot_hypertension(dataset_in):
    # קיבוץ לפי לחץ דם גבוה ושבץ מוחי
    grouped = (
        dataset_in.groupby(["hypertension", "stroke"]).size().unstack(fill_value=0)
    )

    # שינוי שמות לעברית
    grouped.index = ["No Hypertension", "Yes Hypertension"]
    grouped.columns = ["No", "Yes"]

    # ציור הגרף
    ax = grouped.plot(kind="bar", figsize=(8, 6), color=["skyblue", "salmon"])

    # הוספת כותרות
    plt.title("Status: Hypertension Of The Patients:")
    plt.ylabel("Number Of Patients")
    plt.xticks(rotation=0)
    plt.legend(title="Status - Stroke")
    plt.grid(axis="y")
    plt.tight_layout()

    # הוספת המספרים מעל כל עמודה
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            ax.annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.show()


def top_n_models(scores_df:pd.DataFrame, n=5, sort_by="accuracy"):
    """Display a table of the top n models and their scores."""
    top_models = scores_df.sort_values(by=sort_by, ascending=False).head(n)
    top_models = top_models.round(3) # Round the scores to 2 decimal places
    return top_models


def plot_models_table(model_df: pd.DataFrame):
    """Plot the top models dataframe as a table, coloring best and 2nd best in each float column."""
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(
        figsize=(min(12, 2 + model_df.shape[0]), 0.6 + 0.4 * model_df.shape[0])
    )
    ax.axis("off")

    # Prepare cell colors
    cell_colors = [["white"] * model_df.shape[1] for _ in range(model_df.shape[0])]
    float_cols = model_df.select_dtypes(include=['float', 'float64']).columns

    for col_idx, col in enumerate(model_df.columns):
        if col in float_cols:
            col_values = model_df[col].values
            # Get sorted indices (descending)
            sorted_idx = np.argsort(-col_values)
            if len(sorted_idx) > 0:
                cell_colors[sorted_idx[0]][col_idx] = "#90ee90"  # light green for best
            if len(sorted_idx) > 1:
                cell_colors[sorted_idx[1]][col_idx] = "#add8e6"  # light blue for 2nd best

    table = ax.table(
        cellText=model_df.values.astype(str),
        colLabels=model_df.columns,
        loc="center",
        cellLoc="center",
        cellColours=cell_colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(model_df.columns))))
    plt.tight_layout()
    plt.show()


def plot_stroke_distribution_table(dataset_in:pd.DataFrame):
    """Display the distribution of stroke cases as a table with percentage using matplotlib."""
    stroke_counts = dataset_in["stroke"].value_counts()
    total = stroke_counts.sum()
    percent = (stroke_counts / total * 100).round(2)
    percent = " (" + percent.astype(str) + "%)"
    stroke_counts = stroke_counts.astype(str) + percent
    table_df = pd.DataFrame({
        "Count": stroke_counts,
    })
    table_df.index = ["No Stroke", "Stroke"]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.axis('off')
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        rowLabels=table_df.index,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(table_df.columns))))
    table.scale(1.2, 1.2)
    plt.title("Stroke Distribution in Dataset")
    plt.tight_layout()
    plt.show()
    return table_df

# Example

sns.set_theme(style="ticks")

scores_df = pd.read_csv("scores.csv")

boolean_cols = ['PCA', 'PickBest', 'OverSampling', 'OverUnderSampling', 'UnderSampling', 'oversampling', 'overunderSampling', 'undersampling']
existing_booleans = [col for col in boolean_cols if col in scores_df.columns]
other_cols = [col for col in scores_df.columns if col not in (['Model'] + existing_booleans)]
ordered_cols = ['Model'] + existing_booleans + other_cols
scores_df = scores_df[ordered_cols]


top = top_n_models(scores_df, n=10, sort_by=["recall","f1", "precision","accuracy"])
plot_stroke_distribution_table(dataset_in=dataset)
# plot_models_table(top)


# # plt.figure(figsize=(12, 6))
# plot_stroke_distribution(dataset)

# %%
