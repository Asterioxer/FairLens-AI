from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from data_preprocessing import load_and_preprocess_data

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred

# Test in Colab
X, y, sensitive_data = load_and_preprocess_data()
model, X_test, y_test, y_pred = train_model(X, y)
print("Model trained. Predictions shape:", y_pred.shape)
