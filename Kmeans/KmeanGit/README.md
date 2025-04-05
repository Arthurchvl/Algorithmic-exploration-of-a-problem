# Implémentation du Clustering K-Means en Python

## Documentation

### Attributs

`KMeans(self, n_clusters = 3, tolerance = 0.01, max_iter = 100, runs = 1, init_method="forgy")`

`n_clusters` : Nombre de clusters

`tolerance` : Valeur de tolérance. L'algorithme s'arrête si la distance entre les centroïdes précédents et les centroïdes actuels est inférieure à la tolérance.

`max_iter` : Nombre d'itérations à chaque exécution.

`runs` : Détermine combien de fois l'algorithme sera exécuté. Cela a du sens uniquement si une méthode d'initialisation aléatoire est utilisée. Par conséquent, cette option est ignorée lorsqu'une méthode d'initialisation non aléatoire est choisie.

`init_method` : Méthode d'initialisation. Seules quatre méthodes sont implémentées : Forgy, Macqueen, Maximin, Var-Part.

Macqueen sélectionne simplement les K premières observations et les assigne comme centroïdes.

Forgy prend K points de données aléatoires comme centroïdes initiaux.

Maximin et Var-Part sont des méthodes d'initialisation plus sophistiquées. Var-Part est généralement plus efficace. Les liens vers les articles correspondants sont dans la section ci-dessous.

`KMeans.fit(X)`: Exécute l'algorithme K-Means.


## Références

#### Comparaison générale des différentes méthodes d'initialisation  
Une étude comparative des méthodes d'initialisation efficaces pour l'algorithme de clustering K-Means : https://arxiv.org/abs/1209.1960

#### Méthodes PCA-Part et Var-Part  
À la recherche de méthodes déterministes pour l'initialisation du K-Means et du clustering par mélange gaussien : https://www.researchgate.net/publication/220571343_In_search_of_deterministic_methods_for_initializing_K-means_and_Gaussian_mixture_clustering

#### Méthode Maximin  
Une nouvelle technique d'initialisation pour l'itération généralisée de Lloyd : https://ieeexplore.ieee.org/document/329844