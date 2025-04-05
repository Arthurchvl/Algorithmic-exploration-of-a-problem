# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

class KMeansClustering:
    def __init__(self, n_clusters: int = 5, max_iter: int = 300, random_state: int = 0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.inertia_ = 0
        
    def initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initialize centroids using k-means++ method."""
        np.random.seed(self.random_state)

        centroids = [X[np.random.randint(0, X.shape[0])]]
        
        for _ in range(1, self.n_clusters):
            distances = np.array([min([np.linalg.norm(x - c) for c in centroids]) for x in X])
            probs = distances**2 / np.sum(distances**2)
            cumulative_probs = np.cumsum(probs)
            r = random.random()
            idx = np.where(cumulative_probs >= r)[0][0]
            centroids.append(X[idx])
        
        return np.array(centroids)
    
    def assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign clusters based on the nearest centroid."""
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.linalg.norm(X - centroids[k], axis=1)
        return np.argmin(distances, axis=1)
    
    def update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Update centroids based on the assigned clusters."""
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            if np.sum(labels == k) > 0:
                centroids[k] = np.mean(X[labels == k], axis=0)
        return centroids
    
    def calculate_inertia(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """Calculate the inertia (sum of squared distances to closest centroid)."""
        inertia = 0
        for k in range(self.n_clusters):
            if np.sum(labels == k) > 0:
                inertia += np.sum(np.linalg.norm(X[labels == k] - centroids[k], axis=1)**2)
        return inertia
    
    def fit(self, X: np.ndarray) -> np.ndarray:
        """Fit the model to the data."""
        self.cluster_centers_ = self.initialize_centroids(X)
        
        for _ in range(self.max_iter):
            labels = self.assign_clusters(X, self.cluster_centers_)
            new_centroids = self.update_centroids(X, labels)
            
            if np.all(self.cluster_centers_ == new_centroids):
                break
                
            self.cluster_centers_ = new_centroids
        
        labels = self.assign_clusters(X, self.cluster_centers_)
        self.inertia_ = self.calculate_inertia(X, labels, self.cluster_centers_)
        
        return labels
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit the model and predict the cluster labels."""
        return self.fit(X)

def load_data(file_path: str) -> np.ndarray:
    """Load dataset from a CSV file."""
    try:
        dataset = pd.read_csv(file_path)
        return dataset.iloc[:, [3, 4]].values
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

def plot_clusters(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> None:
    """Visualize the clusters and centroids."""
    plt.figure(figsize=(10, 6))
    for i in range(np.max(labels) + 1):
        plt.scatter(X[labels == i, 0], X[labels == i, 1], s=100, label=f'Cluster {i + 1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='yellow', label='Centroids')
    plt.title('Clusters of clients', fontsize=16)
    plt.xlabel('Annual Income (k$)', fontsize=14)
    plt.ylabel('Spending score (1-100)', fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()

def main(file_path: str) -> None:
    """Main function to execute K-Means clustering."""
    X = load_data(file_path)
    
    # Using the elbow method to find the optimal number of clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeansClustering(n_clusters=i, max_iter=300, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
        
    # Applying KMeans to the dataset with the optimal number of clusters
    optimal_clusters = 5  # This can be adjusted based on the elbow method result
    kmeans = KMeansClustering(n_clusters=optimal_clusters, max_iter=300, random_state=0)
    y_kmeans = kmeans.fit_predict(X)
    
    plot_clusters(X, y_kmeans, kmeans.cluster_centers_)
    

if __name__ == "__main__":
    main('Mall_Customers.csv')