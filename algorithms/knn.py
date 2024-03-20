from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time


def train_knn(X_train, y_train, n_neighbors=3):
    """
    Train a K-Nearest Neighbor model with the training data.
    """
    # Initialize the KNN classifier with n neighbors
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Record the training start time
    start_time = time.time()

    # Train the model
    model.fit(X_train, y_train)

    # Calculate the training duration
    training_duration = time.time() - start_time

    return model, training_duration

# The evaluate_model function defined in naive_bayes.py can be reused here
