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


    def initialiser_centroides(self, X: np.ndarray) -> np.ndarray:
        """Initialiser les centroïdes en utilisant la méthode k-means++."""
        np.random.seed(self.random_state)

        centroides = [X[np.random.randint(0, X.shape[0])]]
        
        for _ in range(1, self.n_clusters):
            distances = np.array([min([np.linalg.norm(x - c) for c in centroides]) for x in X])
            probs = distances**2 / np.sum(distances**2)
            cumulative_probs = np.cumsum(probs)
            r = random.random()
            idx = np.where(cumulative_probs >= r)[0][0]
            centroides.append(X[idx])
        
        return np.array(centroides)


    def affecter_clusters(self, X: np.ndarray, centroides: np.ndarray) -> np.ndarray:
        """Affecter les clusters en fonction du centreide le plus proche."""
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.linalg.norm(X - centroides[k], axis=1)
        return np.argmin(distances, axis=1)


    def mettre_a_jour_centroides(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Mettre à jour les centroïdes en fonction des clusters assignés."""
        centroides = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            if np.sum(labels == k) > 0:
                centroides[k] = np.mean(X[labels == k], axis=0)
        return centroides


    def calculer_inertie(self, X: np.ndarray, labels: np.ndarray, centroides: np.ndarray) -> float:
        """Calculer l'inertie (somme des distances au carré du centreide le plus proche)."""
        inertie = 0
        for k in range(self.n_clusters):
            if np.sum(labels == k) > 0:
                inertie += np.sum(np.linalg.norm(X[labels == k] - centroides[k], axis=1)**2)
        return inertie


    def ajuster(self, X: np.ndarray) -> np.ndarray:
        """Ajuster le modèle aux données."""
        self.cluster_centers_ = self.initialiser_centroides(X)
        
        for _ in range(self.max_iter):
            labels = self.affecter_clusters(X, self.cluster_centers_)
            nouveaux_centroides = self.mettre_a_jour_centroides(X, labels)
            
            if np.all(self.cluster_centers_ == nouveaux_centroides):
                break
                
            self.cluster_centers_ = nouveaux_centroides
        
        labels = self.affecter_clusters(X, self.cluster_centers_)
        self.inertia_ = self.calculer_inertie(X, labels, self.cluster_centers_)
        
        return labels
    
    def ajuster_predire(self, X: np.ndarray) -> np.ndarray:
        """Ajuster le modèle et prédire les labels des clusters."""
        return self.ajuster(X)
    

def charger_donnees(chemin_fichier: str) -> np.ndarray:
    """Charger le jeu de données à partir d'un fichier CSV."""
    try:
        dataset = pd.read_csv(chemin_fichier)
        return dataset.iloc[:, [3, 4]].values
    except FileNotFoundError:
        print(f"Erreur : Le fichier {chemin_fichier} n'a pas été trouvé.")
        raise
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")
        raise


def afficher_clusters(X: np.ndarray, labels: np.ndarray, centroides: np.ndarray) -> None:
    """Visualiser les clusters et les centroïdes."""
    plt.figure(figsize=(10, 6))
    for i in range(np.max(labels) + 1):
        plt.scatter(X[labels == i, 0], X[labels == i, 1], s=100, label=f'Cluster {i + 1}')
    plt.scatter(centroides[:, 0], centroides[:, 1], s=300, c='yellow', label='Centroïdes')
    plt.title('Clusters des clients', fontsize=16)
    plt.xlabel('Revenu Annuel (k$)', fontsize=14)
    plt.ylabel('Score de dépense (1-100)', fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


def principal(chemin_fichier: str) -> None:
    """Fonction principale pour exécuter le clustering K-Means."""
    X = charger_donnees(chemin_fichier)
    
    # Utilisation de la méthode du coude pour trouver le nombre optimal de clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeansClustering(n_clusters=i, max_iter=300, random_state=0)
        kmeans.ajuster(X)
        wcss.append(kmeans.inertia_)
        
    # Application de KMeans au jeu de données avec le nombre optimal de clusters
    clusters_optimaux = 5  # Cela peut être ajusté en fonction du résultat de la méthode du coude
    kmeans = KMeansClustering(n_clusters=clusters_optimaux, max_iter=300, random_state=0)
    y_kmeans = kmeans.ajuster_predire(X)
    
    afficher_clusters(X, y_kmeans, kmeans.cluster_centers_)

if __name__ == "__main__":
    principal('Mall_Customers.csv')
