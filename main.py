# main.py

import logging
from src.data_preprocessing import load_data, preprocess_data, split_data, scale_data
from src.eda import perform_eda
from src.model_training import train_logistic_regression, train_random_forest
from src.evaluation import evaluate_model, cross_validation
from src.visualizations import plot_loan_approval, plot_distributions, plot_correlation, plot_confusion_matrix
from sklearn.model_selection import KFold
import pandas as pd

def setup_logging():
    logging.basicConfig(filename='logs/app.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

def main():
    setup_logging()
    try:
        # Load Data
        df = load_data('H:/My Drive/BISI II/Data Science/Term Assignments/Loan_Eligibility_Model_Solution/data/credit.csv')
        df = df.drop(columns=['Loan_ID'])
        df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'])
        # Perform EDA
        perform_eda(df)
        plot_loan_approval(df)
        plot_distributions(df)
        plot_correlation(df)
        
        # Preprocess Data
        df = preprocess_data(df)
        
        # Print head of dataframe to debug
        print("Head of DataFrame after preprocessing: ")
        print(df.head())
        
        # Split Data
        xtrain, xtest, ytrain, ytest = split_data(df)
        
        # Scale Data
        xtrain_scaled, xtest_scaled = scale_data(xtrain, xtest)
        
        # Train Models
        lrmodel = train_logistic_regression(xtrain_scaled, ytrain)
        rfmodel = train_random_forest(xtrain, ytrain)
        
        # Evaluate Models
        print("Logistic Regression Evaluation")
        ypred_lr = lrmodel.predict(xtest_scaled)
        evaluate_model(ytest, ypred_lr)
        plot_confusion_matrix(ytest, ypred_lr)
        
        print("Random Forest Evaluation")
        ypred_rf = rfmodel.predict(xtest)
        evaluate_model(ytest, ypred_rf)
        plot_confusion_matrix(ytest, ypred_rf)
        
        # Cross Validation
        kfold = KFold(n_splits=5)
        print("Logistic Regression Cross Validation")
        cross_validation(lrmodel, xtrain_scaled, ytrain, kfold)
        
        print("Random Forest Cross Validation")
        cross_validation(rfmodel, xtrain, ytrain, kfold)
        
    except Exception as e:
        logging.error("An error occurred: %s", str(e))
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()