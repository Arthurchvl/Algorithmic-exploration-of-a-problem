from csv import reader
from sys import exit
from math import sqrt
from operator import itemgetter
from chebyshev import calculChebyshevDistance
from manhattan import calculManhattanDistance
from euclidienne import calculEuclidianDistance


def charger_ensemble_donnees(nom_fichier):
    try:
        with open(nom_fichier, newline='') as iris:
            return list(reader(iris, delimiter=','))
    except FileNotFoundError as e:
        raise e


def convertir_en_flottant(ensemble_donnees, mode):
    nouvel_ensemble = []
    try:
        if mode == 'entrainement':
            for donnees in ensemble_donnees:
                nouvel_ensemble.append([float(x) for x in donnees[:len(donnees)-1]] + [donnees[len(donnees)-1]])

        elif mode == 'test':
            for donnees in ensemble_donnees:
                nouvel_ensemble.append([float(x) for x in donnees])

        else:
            print("Mode invalide, le programme va s'arrêter.")
            exit()

        return nouvel_ensemble

    except ValueError as v:
        print(v)
        print("Format d'ensemble de données invalide, le programme va s'arrêter.")
        exit()


def obtenir_classes(ensemble_entrainement):
    return list(set([c[-1] for c in ensemble_entrainement]))


def trouver_voisins(distances, k):
    return distances[0:k]


def determiner_reponse(voisins, classes):
    votes = [0] * len(classes)

    for instance in voisins:
        for ctr, c in enumerate(classes):
            if instance[-2] == c:
                votes[ctr] += 1

    return max(enumerate(votes), key=itemgetter(1))


def calculer_distance(point1, point2, type_distance):
    """
    Calcule la distance entre deux points en utilisant la métrique spécifiée
    """
    # Convertir les listes en tuples pour compatibilité avec les fonctions de distance
    point1_tuple = tuple(point1)
    point2_tuple = tuple(point2)
    
    if type_distance == 'euclidienne':
        return calculEuclidianDistance(point1_tuple, point2_tuple)

    elif type_distance == 'manhattan':
        return calculManhattanDistance(point1_tuple, point2_tuple)
    
    elif type_distance == 'chebyshev':
        return calculChebyshevDistance(point1_tuple, point2_tuple)
    
    else:
        print("Type de distance invalide, utilisation par défaut de la distance Euclidienne")
        return calculEuclidianDistance(point1, point2)


def knn(ensemble_entrainement, ensemble_test, k, type_distance):
    distances = []
    limite = len(ensemble_entrainement[0]) - 1

    # Générer les classes à partir des données d'entraînement
    classes = obtenir_classes(ensemble_entrainement)

    try:
        for instance_test in ensemble_test:
            for ligne in ensemble_entrainement:
                # Calculer la distance en utilisant la métrique sélectionnée
                dist = calculer_distance(ligne[:limite], instance_test, type_distance)
                distances.append(ligne + [dist])

            distances.sort(key=itemgetter(len(distances[0])-1))

            # Trouver les k plus proches voisins
            voisins = trouver_voisins(distances, k)

            # Déterminer la classe avec le maximum de votes
            index, valeur = determiner_reponse(voisins, classes)

            # Afficher la prédiction
            print("La classe prédite pour l'échantillon " + str(instance_test) + " est : " + classes[index])
            print("Nombre de votes : " + str(valeur) + " sur " + str(k))
            print("Métrique de distance utilisée : " + type_distance.capitalize())
            print('-' * 50)

            # Vider la liste des distances
            distances.clear()

    except Exception as e:
        print(e)


def principal():
    try:
        # Obtenir la valeur de k
        k = int(input("Entrez la valeur de k : "))

        # Demander à l'utilisateur de choisir une métrique de distance
        print("\nMétriques de distance disponibles:")
        print("\n1. Distance Euclidienne (cercle)")
        print("2. Distance Manhattan (losange)")
        print("3. Distance Chebyshev (carré)")
        
        choix = int(input("\nChoisissez une métrique de distance (1-3) : "))
        
        if choix == 1:
            type_distance = 'euclidienne'
        elif choix == 2:
            type_distance = 'manhattan'
        elif choix == 3:
            type_distance = 'chebyshev'
        else:
            print("Choix invalide, utilisation par défaut de la distance Euclidienne")
            type_distance = 'euclidienne'
        
        # Charger les ensembles de données d'entraînement et de test
        fichier_entrainement = input("\nEntrez le nom du fichier de données d'entraînement : ")
        fichier_test = input("\nEntrez le nom du fichier de données de test : ")
        ensemble_entrainement = convertir_en_flottant(charger_ensemble_donnees(fichier_entrainement), 'entrainement')
        ensemble_test = convertir_en_flottant(charger_ensemble_donnees(fichier_test), 'test')

        if not ensemble_entrainement:
            print("Ensemble d'entraînement vide")

        elif not ensemble_test:
            print("Ensemble de test vide")

        elif k > len(ensemble_entrainement):
            print("Le nombre attendu de voisins est supérieur au nombre d'échantillons d'entraînement")

        else:
            print(f"\nExécution de KNN avec la métrique {type_distance.capitalize()}...\n")
            knn(ensemble_entrainement, ensemble_test, k, type_distance)

    except ValueError as v:
        print(v)

    except FileNotFoundError:
        print("Fichier introuvable")


if __name__ == '__main__':
    principal()