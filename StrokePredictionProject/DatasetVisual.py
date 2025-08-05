from matplotlib.patches import Patch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import ConfusionMatrixDisplay

# Suppress all warnings
warnings.filterwarnings("ignore")

# Read from the dataset
dataset = pd.read_csv("healthcare-dataset-stroke-data.csv")


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


def plot_pca_explained_var(dataset_in: pd.DataFrame):
    # Dimension reduction
    # Create scaler
    scaler = StandardScaler()
    # Create a PCA instance
    pca = PCA()
    # Create pipeline
    pipeline = make_pipeline(scaler, pca)

    x = dataset_in.drop("stroke", axis=1)

    # Fit the pipeline to samples
    pipeline.fit(x)
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    # Plot the explained variances
    features = range(pca.n_components_)
    plt.plot(cumulative_variance_ratio)
    plt.xlabel("Number of principal components")
    plt.ylabel("cumulative explained variance ratio")
    plt.xticks(features)
    plt.title("Cumulative Explained Variance Ratio by Principal Components")
    plt.show()


def plot_bmi_distribution(df: pd.DataFrame):
    column = "bmi"
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111)
    sns.kdeplot(
        data=df[df["stroke"] == 0],
        x=column,
        ax=ax1,
        shade=True,
        alpha=1,
        color="salmon",
    )
    kax = sns.kdeplot(
        data=df[df["stroke"] == 1],
        x=column,
        ax=ax1,
        shade=True,
        alpha=0.8,
        color="skyblue",
    )
    kax.legend(
        ["False", "True"],
        loc="upper right",
        fontsize=12,
        title="Stroke Status",
        title_fontsize=14,
        framealpha=0.8,
        facecolor="white",
    )
    kax.set_xlabel(column.capitalize(), fontsize=12, labelpad=2)
    kax.set_ylabel("Density", fontsize=12, labelpad=2)
    kax.set_title("Bmi Distribution by Stroke amount", fontsize=16, pad=10)
    plt.tight_layout(pad=2)
    plt.show()


def plot_age_distribution(df: pd.DataFrame):
    column = "age"
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111)
    sns.kdeplot(
        data=df[df["stroke"] == 0],
        x=column,
        ax=ax1,
        shade=True,
        alpha=1,
        color="salmon",
    )
    kax = sns.kdeplot(
        data=df[df["stroke"] == 1],
        x=column,
        ax=ax1,
        shade=True,
        alpha=0.8,
        color="skyblue",
    )
    kax.legend(
        ["False", "True"],
        loc="upper right",
        fontsize=12,
        title="Stroke Status",
        title_fontsize=14,
        framealpha=0.8,
        facecolor="white",
    )
    kax.set_xlabel("Age", fontsize=12, labelpad=2)
    kax.set_ylabel("Density", fontsize=12, labelpad=2)
    kax.set_title("Age Distribution by Stroke amount", fontsize=16, pad=10)
    plt.tight_layout(pad=2)
    plt.show()


def plot_glucose_distribution(df: pd.DataFrame):
    column = "avg_glucose_level"
    fig = plt.figure("glucose_distiribution", figsize=(12, 6))
    ax1 = fig.add_subplot(111)
    sns.kdeplot(
        data=df[df["stroke"] == 0],
        x=column,
        ax=ax1,
        shade=True,
        alpha=1,
        color="salmon",
    )
    kax = sns.kdeplot(
        data=df[df["stroke"] == 1],
        x=column,
        ax=ax1,
        shade=True,
        alpha=0.8,
        color="skyblue",
    )
    kax.legend(
        ["False", "True"],
        loc="upper right",
        fontsize=12,
        title="Stroke Status",
        title_fontsize=14,
        framealpha=0.8,
        facecolor="white",
    )
    kax.set_xlabel("Average glucose level", fontsize=12, labelpad=2)
    kax.set_ylabel("Density", fontsize=12, labelpad=2)
    kax.set_title("Glucose level Distribution by Stroke amount", fontsize=16, pad=10)
    plt.tight_layout(pad=2)
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


def top_n_models(scores_df: pd.DataFrame, n=5, by="accuracy"):
    """Display a table of the top n models and their scores."""
    top_models = scores_df.sort_values(by=by, ascending=False).head(n)
    top_models = top_models.round(3)  # Round the scores to 2 decimal places
    return top_models


def plot_models_table(
    model_df: pd.DataFrame,
    with_index: bool = False,
    index_name: str = "Index",
    title: str = "",
):
    """Plot the top models dataframe as a beautiful table, coloring best and 2nd best in each float column."""
    import matplotlib.colors as mcolors

    # Use two colors from the same palette for best and 2nd best
    blues = sns.color_palette("Greens", 8)
    color_default = "#e6f1fd"
    color_best = mcolors.to_hex(blues[4])
    color_2ndBest = mcolors.to_hex(blues[1])
    color_header = "#d2e6fa"

    # Prepare cell colors
    n_rows, n_cols = model_df.shape
    cell_colors = []
    for i in range(n_rows):
        row_color = color_default
        cell_colors.append([row_color] * n_cols)

    float_cols = model_df.select_dtypes(include=["float", "float64"]).columns

    for col_idx, col in enumerate(model_df.columns):
        if col in float_cols:
            col_values = model_df[col].values
            sorted_idx = np.argsort(-col_values)
            if len(sorted_idx) > 0:
                cell_colors[sorted_idx[0]][col_idx] = color_best
            if len(sorted_idx) > 1:
                cell_colors[sorted_idx[1]][col_idx] = color_2ndBest

    if with_index:
        for i, row in enumerate(cell_colors):
            row.insert(0, color_default)
        cellText = np.column_stack(
            [model_df.index.astype(str), model_df.values.astype(str)]
        )
        colLabels = [index_name] + list(model_df.columns)
        header_colors = [color_header] * (n_cols + 1)
    else:
        cellText = model_df.values.astype(str)
        colLabels = list(model_df.columns)
        header_colors = [color_header] * n_cols

    fig, ax = plt.subplots(
        figsize=(
            min(14, 2 + model_df.shape[1] + (with_index)),
            0.8 + 0.5 * (model_df.shape[0] + (with_index)),
        )
    )
    ax = fig.axes[0]
    ax.axis("off")
    # Add table
    table = ax.table(
        cellText=cellText,
        colLabels=colLabels,
        loc="center",
        cellLoc="center",
        cellColours=cell_colors,
        colColours=header_colors,
        rowLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.auto_set_column_width(col=list(range(len(colLabels))))

    # Beautify header
    for key, cell in table.get_celld().items():
        row, col = key
        if row == 0 or col == 0:
            cell.set_text_props(weight="bold", color="#222")
            cell.set_edgecolor("black")
            cell.set_linewidth(1.5)
        else:
            cell.set_edgecolor("black")
            cell.set_linewidth(1.5)
        cell.set_height(0.1)
    fig.suptitle(
        title,
        fontsize=20,
        fontweight="semibold",
        fontstyle="italic",
        y=0.98,  # Adjust vertical position if needed
    )
    plt.tight_layout()
    plt.show()


def filter_df_by_boolean(df: pd.DataFrame, boolean_filter: dict) -> pd.DataFrame:
    """
    Filter the DataFrame based on boolean conditions.

    Parameters:
    df (pd.DataFrame): The DataFrame to filter.
    boolean_filter (dict): A dictionary where keys are column names and values are the boolean values to filter by.

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """
    mask = True
    for key, val in boolean_filter.items():
        mask &= df[key] == val

    return df[mask]


def plot_model_Oversampling_comparison(scores: pd.DataFrame, boolean_filter: dict):
    """Plot a comparison of models with and without oversampling."""
    boolean_cols_woOverSamp = ["PCA", "PickBest", "OverUnderSampling", "UnderSampling"]
    df_temp = filter_df_by_boolean(scores, boolean_filter)
    df_oversmapling_comp = (
        df_temp.drop(labels=boolean_cols_woOverSamp, axis=1)
        .sort_values(["f1", "recall"], ascending=False)
        .groupby(["Model", "OverSampling"])
        .head(n=1)
        .round(3)
        .sort_values(["Model", "OverSampling"], ascending=False)
    )
    df_oversmapling_comp["OverSampling"].replace([True], "With", inplace=True)
    df_oversmapling_comp["OverSampling"].replace([False], "Without", inplace=True)

    plot_models_table(
        df_oversmapling_comp,
        False,
        "Model",
        "Comparing the impact of Oversampling on each model",
    )


def plot_Model_PCA_comparison(scores: pd.DataFrame, boolean_filter: dict):
    """Plot a comparison of models with and without PCA."""

    boolean_cols_woPCA = [
        "PickBest",
        "OverSampling",
        "OverUnderSampling",
        "UnderSampling",
    ]
    df_temp = filter_df_by_boolean(scores, boolean_filter)
    df_PCA_comp = (
        df_temp.drop(labels=boolean_cols_woPCA, axis=1)
        .sort_values(["f1", "recall"], ascending=False)
        .groupby(["Model", "PCA"])
        .head(n=1)
        .round(3)
        .sort_values(["Model", "PCA"], ascending=False)
    )

    df_PCA_comp["PCA"].replace([True], "With", inplace=True)
    df_PCA_comp["PCA"].replace([False], "Without", inplace=True)

    plot_models_table(
        df_PCA_comp, False, "Model", "Comparing the impact of PCA on each model"
    )


def plotModelScoresComparison(
    scores_df: pd.DataFrame, boolean_filter: dict, score_metrics
):
    boolean_cols = [
        "PCA",
        "PickBest",
        "OverSampling",
        "OverUnderSampling",
        "UnderSampling",
    ]

    ## reorder columns to have Model name first and scores last (most right)
    existing_booleans = [col for col in boolean_cols if col in scores_df.columns]
    other_cols = [
        col for col in scores_df.columns if col not in (["Model"] + existing_booleans)
    ]
    ordered_cols = ["Model"] + existing_booleans + other_cols
    scores_df = scores_df[ordered_cols]
    # top = top_n_models(scores_df, n=30, by=score_metrics)
    df = (
        filter_df_by_boolean(scores_df, bool_filter)
        .drop(labels=boolean_cols, axis=1)
        .sort_values(["f1"], ascending=False)
    )
    df = df.groupby("Model").head(1).round(3)

    sns.set_theme(style="darkgrid")

    mdl_cmp_title = "Comparing between each model"

    # plot_glucose_distribution(df=dataset)
    # plot_Model_PCA_comparison(scores_df, bool_filter)
    plot_models_table(df, False, "Model", mdl_cmp_title)


def plotAverageConfusionmatrices(
    confusion_matrices: dict,
    models: dict,
):
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    for ax, (name, cm) in zip(axes, confusion_matrices.items()):
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(name)
    fig.suptitle("Confusion Matrix for Each Model", fontsize=16)
    plt.tight_layout()
    plt.show()


scores_df = pd.read_csv("scores.csv")

# filter items on df by which method was added for model
bool_filter = {
    "PCA": False,
    "PickBest": False,
    "OverSampling": False,
    "OverUnderSampling": False,
    "UnderSampling": False,
}
boolean_cols = ["PCA", "PickBest", "OverSampling", "OverUnderSampling", "UnderSampling"]
score_metrics = ["recall", "f1", "precision", "accuracy"]

## reorder columns to have Model name first and scores last (most right)
existing_booleans = [col for col in boolean_cols if col in scores_df.columns]
other_cols = [
    col for col in scores_df.columns if col not in (["Model"] + existing_booleans)
]
ordered_cols = ["Model"] + existing_booleans + other_cols
scores_df = scores_df[ordered_cols]
# top = top_n_models(scores_df, n=30, by=score_metrics)
df = (
    filter_df_by_boolean(scores_df, bool_filter)
    .drop(labels=boolean_cols, axis=1)
    .sort_values(["f1"], ascending=False)
)
df = df.groupby("Model").head(1).round(3)

sns.set_theme(style="darkgrid")

mdl_cmp_title = "Comparing between each model"
pca_cmp_title = "Comparing impact of PCA on each model"

# plot_glucose_distribution(df=dataset)
# plot_Model_PCA_comparison(scores_df, bool_filter)
# plot_models_table(df, False, "Model", mdl_cmp_title)
# plot_pca_explained_var(dataset)
# # plt.figure(figsize=(12, 6))
# plot_stroke_distribution(dataset)
