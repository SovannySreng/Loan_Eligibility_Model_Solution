from src.data_preprocessing import load_data, preprocess_data
from src.eda import eda
from src.feature_engineering import feature_engineering
from src.model_training import train_model
from src.evaluation import evaluate_model
from src.visualizations import plot_histograms, plot_categorical_distribution
from src.utils import setup_logging, log_error
from sklearn.model_selection import train_test_split

def main():
    setup_logging()
    
    try:
        df = load_data('H:\My Drive\BISI II\Data Science\Term Assignments\Loan_Eligibility_Model_Solution\data\credit.csv')
        
        # Perform EDA
        eda(df)
        
        # Preprocess Data
        df = preprocess_data(df)
        
        # Feature Engineering
        df = feature_engineering(df)
        
        # Visualizations
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        plot_histograms(df, num_cols)
        plot_categorical_distribution(df, cat_cols)
        
        # Split the data
        X = df.drop('target', axis=1)
        y = df['target']
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = train_model(x_train, y_train)
        
        # Evaluate the model
        evaluate_model(model, x_test, y_test)
        
    except Exception as e:
        log_error(e)
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()