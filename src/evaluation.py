

# src/evaluation.py

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model(y_true, y_pred):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

def cross_validation(model, xtrain, ytrain, kfold):
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, xtrain, ytrain, cv=kfold)
    print("Accuracy scores:", scores)
    print("Mean accuracy:", scores.mean())
    print("Standard deviation:", scores.std())