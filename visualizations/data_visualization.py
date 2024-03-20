import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np


def visualize_data():
    """
    Visualize the Iris dataset using a pair plot to show the distribution of each class.
    """
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names

    # Create a DataFrame with the features and labels
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = np.array([iris.target_names[label] for label in y])

    # Plot the pairplot
    sns.pairplot(df, hue="species", markers=["o", "s", "D"])
    plt.title("Iris Data Distribution by Species")
    plt.show()


if __name__ == "__main__":
    visualize_data()
