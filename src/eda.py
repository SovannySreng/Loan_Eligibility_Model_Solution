# src/eda.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def perform_eda(df):
    print("Shape of the dataframe:", df.shape)
    print("Data types of columns:\n", df.dtypes)
    print("Number of missing values in each column:\n", df.isnull().sum())
    print("Summary statistics of numerical columns:\n", df.describe())

def plot_loan_approval(df):
    plt.figure(figsize=(8, 6))
    df['Loan_Approved'].value_counts().plot.bar()
    plt.title("Loan Approval Status")
    plt.xlabel("Status")
    plt.ylabel("Count")
    plt.show()

def plot_distributions(df):
    plt.figure(figsize=(14, 14))
    sns.distplot(df['LoanAmount'])
    plt.title("Loan Amount Distribution")
    plt.show()

def plot_correlation(df):
    plt.figure(figsize=(15, 8))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='YlGnBu')
    plt.title("Correlation Matrix")
    plt.show()