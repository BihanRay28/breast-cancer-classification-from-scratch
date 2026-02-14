import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(test_size=0.2, random_state=42):
    # Load dataset
    data = load_breast_cancer()
    X = data.data          # shape (m, n_x)
    Y = data.target        # shape (m,)

    # Train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    # Normalize features (VERY IMPORTANT for NN)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Transpose to match your NN shape convention
    X_train = X_train.T
    X_test = X_test.T

    Y_train = Y_train.reshape(1, -1)
    Y_test = Y_test.reshape(1, -1)

    return X_train, Y_train, X_test, Y_test
