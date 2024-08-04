from sklearn.linear_model import LogisticRegression

def train_model(x_train, y_train):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    return model