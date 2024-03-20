from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_data(test_size=0.2):
    """
    Load the Iris dataset and split it into train and test sets.
    """
    # Load Iris dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test
