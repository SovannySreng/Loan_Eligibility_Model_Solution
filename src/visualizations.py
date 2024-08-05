
# src/visualizations.py

import seaborn as sns
import matplotlib.pyplot as plt

def plot_loan_approval(df):
    df['Loan_Status'].value_counts().plot.bar()
    
    plt.show()

def plot_distributions(df):
    sns.displot(df['LoanAmount'])
    plt.show()

def plot_correlation(df):
    plt.figure(figsize=(15,8))
    sns.heatmap(df.corr(),annot=True, fmt='0.2f', cmap='YlGnBu')
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()