from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import time


def train_naive_bayes(X_train, y_train):
    """
    Train a Gaussian Naive Bayes model with the training data.
    """
    # Initialize the classifier
    model = GaussianNB()

    # Record the training start time
    start_time = time.time()

    # Train the model
    model.fit(X_train, y_train)

    # Calculate the training duration
    training_duration = time.time() - start_time

    return model, training_duration


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the given model using the test data.
    """
    # Predict the labels for the test data
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the predictions
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy
