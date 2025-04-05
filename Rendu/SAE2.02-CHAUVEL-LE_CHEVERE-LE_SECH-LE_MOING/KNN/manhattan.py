import numpy as np
import matplotlib.pyplot as plt

def calculManhattanDistance(point1, point2):
    #calcul de la distance entre 2 points avec la distance de Manhattan
    
    if not isinstance(point1, tuple) or not isinstance(point2, tuple):
        raise TypeError("Les points doivent être des tuples de nombres.")

    if len(point1) != len(point2):
        raise ValueError("Les points doivent avoir le même nombre de dimensions.")
        
    return sum(abs(p1 - p2) for p1, p2 in zip(point1, point2))

def manhattanDistance():
    # Créer un tableau de 1000 valeurs de x de 1 à -1
    x_values = np.linspace(1, -1, 1000)
    
    # Calculer les valeurs correspondantes de y en utilisant la distance de Manhattan
    y_positive = 1 - np.abs(x_values)  # Partie positive de y
    y_negative = -(1 - np.abs(x_values))  # Partie négative de y
    
    # Créer un graphique
    plt.figure(figsize=(6,6))
    plt.plot(x_values, y_positive, label='y positif')
    plt.plot(x_values, y_negative, label='y négatif')
    
    # Ajouter des étiquettes et une légende
    plt.axhline(0, color='black',linewidth=1)
    plt.axvline(0, color='black',linewidth=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Points à distance de Manhattan égale à 1 par rapport à l\'origine')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Afficher la grille
    plt.grid(True)
    
    # Afficher le graphique
    plt.legend()
    plt.show()