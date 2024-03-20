import matplotlib

from algorithms.naive_bayes import train_naive_bayes, evaluate_model
from algorithms.knn import train_knn
from utils.data_loader import load_data
from visualizations.data_visualization import visualize_data
matplotlib.use('TkAgg')

def main():
    # Load and visualize the data
    X_train, X_test, y_train, y_test = load_data()
    visualize_data()

    # Train and evaluate NB
    nb_model, nb_training_time = train_naive_bayes(X_train, y_train)
    nb_accuracy = evaluate_model(nb_model, X_test, y_test)
    print(f"Naive Bayes Accuracy: {nb_accuracy * 100:.2f}%, Training Time: {nb_training_time:.4f} seconds")

    # Train and evaluate KNN
    knn_model, knn_training_time = train_knn(X_train, y_train)
    knn_accuracy = evaluate_model(knn_model, X_test, y_test)
    print(f"KNN Accuracy: {knn_accuracy * 100:.2f}%, Training Time: {knn_training_time:.4f} seconds")


if __name__ == "__main__":
    main()

