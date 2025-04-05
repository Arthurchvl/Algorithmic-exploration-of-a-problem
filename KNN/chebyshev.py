#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:00:19 2025

@author: arthurchauvel
"""

import numpy as np
import matplotlib.pyplot as plt 

def calculChebyshevDistance(point1, point2):
    #calcul de la distance entre 2 points avec la distance de Chebyshev
    
    if not isinstance(point1, tuple) or not isinstance(point2, tuple):
        raise TypeError("Les points doivent être des tuples de nombres.")

    if len(point1) != len(point2):
        raise ValueError("Les points doivent avoir le même nombre de dimensions.")
        
    return max(abs(p1 - p2) for p1, p2 in zip(point1, point2))

def chebyshevDistance():
    # Créer un tableau de 1000 valeurs de x de 1 à -1
    x_values = np.linspace(1, -1, 1000)
    
    # Calculer les valeurs correspondantes de y en utilisant la distance de Chebyshev
    y_positive = np.ones_like(x_values)  # Partie supérieure du carré (y = 1)
    y_negative = -np.ones_like(x_values)  # Partie inférieure du carré (y = -1)
    
    # Côtés verticaux du carré (x = 1 et x = -1)
    x_vertical = np.ones_like(x_values)  # Côté droit du carré (x = 1)
    x_vertical_neg = -np.ones_like(x_values)  # Côté gauche du carré (x = -1)
    
    # Créer un graphique
    plt.figure(figsize=(6,6))
    
    # Tracer les bords supérieur et inférieur
    plt.plot(x_values, y_positive)
    plt.plot(x_values, y_negative)
    
    # Tracer les bords gauche et droit
    plt.plot(x_vertical, x_values)
    plt.plot(x_vertical_neg, x_values)
    
    # Ajouter des étiquettes et une légende
    plt.axhline(0, color='black',linewidth=1)
    plt.axvline(0, color='black',linewidth=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Points à distance de Chebyshev égale à 1 par rapport à l\'origine')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Afficher la grille
    plt.grid(True)
    
    # Afficher le graphique
    plt.show()