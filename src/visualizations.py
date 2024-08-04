import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_histograms(df: pd.DataFrame, num_cols: list):
    df[num_cols].hist(figsize=(14, 14))
    plt.show()

def plot_categorical_distribution(df: pd.DataFrame, cat_cols: list):
    for col in cat_cols:
        if col != 'target':  # Assuming 'target' is the name of the target column
            pd.crosstab(df[col], df['target'], normalize='index').plot(kind='bar', figsize=(8, 4), stacked=True)
            plt.ylabel('Target Percentage')
            plt.title(f'Distribution of {col} by Target')
            plt.show()