import numpy as np
import math 
import matplotlib.pyplot as plt

def calculEuclidianDistance(point1, point2):
    #calculer la distance entre 2 points à l'aide de la méthode Euclidienne
    if not isinstance(point1, tuple) or not isinstance(point2, tuple):
        raise TypeError("Les points doivent être des tuples de nombres.")

    if len(point1) != len(point2):
        raise ValueError("Les points doivent avoir le même nombre de dimensions.")
    
    return math.sqrt(sum((coord1 - coord2) ** 2 for coord1, coord2 in zip(point1, point2)))

def euclidianDistance():
    # Créer un tableau de 1000 valeurs de x de 1 à -1
    x_values = np.linspace(1, -1, 1000)
    
    # Calculer les valeurs correspondantes de y à partir de l'équation x^2 + y^2 = 1
    y_positive = np.sqrt(1 - x_values**2)  # Partie supérieure du cercle
    y_negative = -np.sqrt(1 - x_values**2)  # Partie inférieure du cercle
    
    # Créer un graphique
    plt.figure(figsize=(6,6))
    plt.plot(x_values, y_positive)
    plt.plot(x_values, y_negative)
    
    # Ajouter des étiquettes et une légende
    plt.axhline(0, color='black',linewidth=2)
    plt.axvline(0, color='black',linewidth=2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Points à distance euclidienne égale à 1 par rapport à l\'origine')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Afficher la grille
    plt.grid(True)
    
    # Afficher le graphique
    plt.legend()
    plt.show()