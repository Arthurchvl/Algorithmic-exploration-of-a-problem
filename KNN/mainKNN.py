from csv import reader
from sys import exit
from operator import itemgetter
from chebyshev import calculChebyshevDistance
from manhattan import calculManhattanDistance
from euclidienne import calculEuclidianDistance


def load_data_set(filename):
    try:
        with open(filename, newline='') as iris:
            return list(reader(iris, delimiter=','))
    except FileNotFoundError as e:
        raise e


def convert_to_float(data_set, mode):
    new_set = []
    try:
        if mode == 'training':
            for data in data_set:
                new_set.append([float(x) for x in data[:len(data)-1]] + [data[len(data)-1]])

        elif mode == 'test':
            for data in data_set:
                new_set.append([float(x) for x in data])

        else:
            print('Invalid mode, program will exit.')
            exit()

        return new_set

    except ValueError as v:
        print(v)
        print('Invalid data set format, program will exit.')
        exit()


def get_classes(training_set):
    return list(set([c[-1] for c in training_set]))


def find_neighbors(distances, k):
    return distances[0:k]


def find_response(neighbors, classes):
    votes = [0] * len(classes)

    for instance in neighbors:
        for ctr, c in enumerate(classes):
            if instance[-2] == c:
                votes[ctr] += 1

    return max(enumerate(votes), key=itemgetter(1))


def calculate_distance(point1, point2, distance_type):
    """
    Calculate distance between two points using the specified distance metric
    """
    # Convert lists to tuples for compatibility with the distance functions
    point1_tuple = tuple(point1)
    point2_tuple = tuple(point2)
    
    if distance_type == 'euclidean':
        return calculEuclidianDistance(point1_tuple, point2_tuple)

    elif distance_type == 'manhattan':
        return calculManhattanDistance(point1_tuple, point2_tuple)
    
    elif distance_type == 'chebyshev':
        return calculChebyshevDistance(point1_tuple, point2_tuple)
    
    else:
        print('Invalid distance type, defaulting to Euclidean')
        calculEuclidianDistance(point1, point2)


def knn(training_set, test_set, k, distance_type):
    distances = []
    limit = len(training_set[0]) - 1

    # generate response classes from training data
    classes = get_classes(training_set)

    try:
        for test_instance in test_set:
            for row in training_set:
                # Calculate distance using selected metric
                dist = calculate_distance(row[:limit], test_instance, distance_type)
                distances.append(row + [dist])

            distances.sort(key=itemgetter(len(distances[0])-1))

            # find k nearest neighbors
            neighbors = find_neighbors(distances, k)

            # get the class with maximum votes
            index, value = find_response(neighbors, classes)

            # Display prediction
            print('The predicted class for sample ' + str(test_instance) + ' is : ' + classes[index])
            print('Number of votes : ' + str(value) + ' out of ' + str(k))
            print('Distance metric used: ' + distance_type.capitalize())
            print('-' * 50)

            # empty the distance list
            distances.clear()

    except Exception as e:
        print(e)


def main():
    try:
        # get value of k
        k = int(input('Enter the value of k : '))

        # Ask user to choose distance metric
        print("\nAvailable distance metrics:")
        print("\n1. Euclidean Distance (circle)")
        print("2. Manhattan Distance (diamond/losange)")
        print("3. Chebyshev Distance (square/carré)")
        
        choice = int(input('\nChoose a distance metric (1-3): '))
        
        if choice == 1:
            distance_type = 'euclidean'
        elif choice == 2:
            distance_type = 'manhattan'
        elif choice == 3:
            distance_type = 'chebyshev'
        else:
            print('Invalid choice, defaulting to Euclidean distance')
            distance_type = 'euclidean'
        
        # load the training and test data set
        training_file = input('\nEnter name of training data file : ')
        test_file = input('\nEnter name of test data file : ')
        training_set = convert_to_float(load_data_set(training_file), 'training')
        test_set = convert_to_float(load_data_set(test_file), 'test')

        if not training_set:
            print('Empty training set')

        elif not test_set:
            print('Empty test set')

        elif k > len(training_set):
            print('Expected number of neighbors is higher than number of training data instances')

        else:
            print(f'\nRunning KNN with {distance_type.capitalize()} distance metric...\n')
            knn(training_set, test_set, k, distance_type)

    except ValueError as v:
        print(v)

    except FileNotFoundError:
        print('File not found')


if __name__ == '__main__':
    main()