# src/data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(file_path='H:/My Drive/BISI II/Data Science/Term Assignments/Loan_Eligibility_Model_Solution/data/credit.csv'):
    return pd.read_csv(file_path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop 'Loan_ID' column
    df = df.drop(columns=['Loan_ID'])
 
        
    # Impute missing values
    df['Gender'].fillna('Male', inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

    # Create dummy variables
    df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'])

    return df

def split_data(df: pd.DataFrame):
    x = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=123)
    return xtrain, xtest, ytrain, ytest

def scale_data(xtrain, xtest):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    xtrain_scaled = scaler.fit_transform(xtrain)
    xtest_scaled = scaler.transform(xtest)
    return xtrain_scaled, xtest_scaled