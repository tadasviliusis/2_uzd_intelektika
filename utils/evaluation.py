from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time


def evaluate_accuracy(model, X_test, y_test):
    """
    Evaluate the accuracy of the model on the test set.
    """
    # Predict the labels for the test data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def evaluate_speed(model, X_train, y_train):
    """
    Measure the time taken to train the model.
    """
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    return training_time


def display_confusion_matrix(y_test, y_pred, class_names):
    """
    Display the confusion matrix for the model predictions.
    """
    matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


def print_classification_report(y_test, y_pred, class_names):
    """
    Print the classification report for the model predictions.
    """
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)
