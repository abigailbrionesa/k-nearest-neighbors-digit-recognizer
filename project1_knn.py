"""
Project 1: k-Nearest Neighbors Digit Recognizer
Abigail Briones

Notes:
- Images are 3x3 grids of pixel values (0-255)
- Labels are integers from 0 to 5
"""

from math import sqrt

def non_empty_list_of_lists(mylist):
    """
    Check if list is a non-empty list of non-empty lists
    """
    if not isinstance(mylist, list) or len(mylist) == 0:
        return False
    for row in mylist:
        if not isinstance(row, list) or len(row) == 0:
            return False
    return True

def non_empty_list(mylist):
    """
    Check if list is a non-empty list
    """
    return isinstance(mylist, list) and len(mylist) > 0

def non_empty_list_of_int(mylist):
    """
    Check if list is a non-empty list of integers
    """
    if not non_empty_list(mylist):
        return False
    for item in mylist:
        if not isinstance(item, int):
            return False
    return True

# Recursive Flattening of Images
def flatten_image(image):
    """
    Recursively flatten a 2D image (list of lists) into a 1D list.

    Precondition:
        - image is a non-empty list of lists
    Postcondition:
        - returns a single list containing all pixel values in row-major order
    """

    if not non_empty_list_of_lists(image):
        print("Image must be a non-empty list of non-empty lists")
        return

    def recursive_flatten(image):
        if not image:
            return []
        return image[0] + recursive_flatten(image[1:])
    
    return recursive_flatten(image)

# Manhattan Distance
def manhattan(image1, image2):
    """
    Compute Manhattan distance between two flattened images.

    Precondition:
        - image1 and image2 are lists of the same length
    Postcondition:
        - returns a non-negative number representing the Manhattan distance
    """
    if not non_empty_list(image1) or not non_empty_list(image2):
        print("Error: Both images must be non-empty lists")
        return None
    if len(image1) != len(image2):
        print("Error: Images must be lists of the same length")
        return None
     
    manhattan_distance = 0
    length = len(image1)

    for i in range(length):
        x1 = image1[i]
        x2 = image2[i]
        if x1 < 0 or x2 < 0:
            print('Error: pixel values must not be negative')
            return None
        manhattan_distance = abs(x2 - x1) + manhattan_distance
    return manhattan_distance

# Euclidean Distance
def euclidean(image1, image2):
    """
    Compute Euclidean distance between two flattened images.

    Precondition:
        - image1 and image2 are lists of the same length
    Postcondition:
        - returns a non-negative number representing the Euclidean distance
    """
    if not non_empty_list(image1) or not non_empty_list(image2):
        print("Error: Both images must be non-empty lists")
        return None
    if len(image1) != len(image2):
        print("Error: Images must be lists of the same length")
        return None

    accumulator = 0
    length = len(image1)

    for i in range(length):
        x1 = image1[i]
        x2 = image2[i]
        if x1 < 0 or x2 < 0:
            print('Error: pixel values must not be negative')
            return None
        accumulator = (x2 - x1) ** 2 + accumulator
    
    euclidean_distance = round(sqrt(accumulator), 3)
    
    return euclidean_distance

# Compute Distances to Training Images
def compute_distances(test_image, training_images, training_labels):
    """
    Compute distances from a test image to all training images.

    Precondition:
        - test_image is a flattened list of pixel values
        - training_images is a list of lists
    Postcondition:
        -  returns list of [label, Manhattan_distance, Euclidean_distance]
    """
    if not non_empty_list_of_int(training_labels):
        print("Error: training_labels must be a non-empty list of integers")
        return None

    distances = []
    length = len(training_images)
    for i in range(length):
        training_image = training_images[i]
        label = training_labels[i]
        manhattan_distance = manhattan(training_image,test_image)
        if manhattan_distance is None:
            return None
        euclidean_distance = euclidean(training_image,test_image)
        if euclidean_distance is None:
            return None
        distances.append([label, manhattan_distance, euclidean_distance])
    
    return distances

# Find k Nearest Neighbors
def k_nearest_neighbors(distances, k):
    """
    Select k training images with the smallest distances for both Manhattan and Euclidean metrics.

    Precondition:
        - distances is a list of lists: [label, Manhattan_distance, Euclidean_distance]
        - k is a positive integer <= number of elements in distances
    Postcondition:
        - returns two lists:
            nearest_neighbors_manhattan: k closest by Manhattan distance
            nearest_neighbors_euclidean: k closest by Euclidean distance
          Each neighbor is a list: [label, Manhattan_distance, Euclidean_distance]
    """
    if not non_empty_list(distances):
        print("Error: distances must be a non-empty list")
        return None
    if not isinstance(k, int) or k <= 0 or k > len(distances):
        print("Error: k must be a positive integer <= number of elements in distances")
        return None
    
    nearest_neighbors_manhattan = []
    nearest_neighbors_euclidean = []

    temp = distances[:]
    for _ in range(k):
        smallest = 0
        for i in range(1,len(temp)):
            if temp[i][1] < temp[smallest][1]:
                    smallest = i
        nearest_neighbors_manhattan.append(temp[smallest])
        temp.pop(smallest)

    temp = distances[:]

    for _ in range(k):
        smallest = 0
        for i in range(1,len(temp)):
            if temp[i][2] < temp[smallest][2]:
                    smallest = i
        nearest_neighbors_euclidean.append(temp[smallest])
        temp.pop(smallest)
            
    print("Manhattan neighbors:", nearest_neighbors_manhattan)
    print("Euclidean neighbors:", nearest_neighbors_euclidean)

    return nearest_neighbors_manhattan, nearest_neighbors_euclidean

# Count Occurrences
def count_occurrences(label, labels):
    """
    Count how many times a specific label appears in a list of labels.

    Precondition:
        - labels is a non-empty list of integers
        - label is an integer
    Postcondition:
        - returns a non-negative integer representing the number of times `label` occurs in `labels`
    """
    if not non_empty_list_of_int(labels):
        print("Error: labels must be a non-empty list of integers")
        return None

    count = 0
    for number in labels:
        if number == label:
            count += 1
    return count

# Majority Voting
def majority_vote(labels):
    """
    Determine predicted label from k nearest neighbors using count_occurrences().

    Precondition:
        - neighbors is a non-empty list of integers
    Postcondition:
        - returns the label with the highest occurrence
        - in case of tie, any one of the top labels may be returned
    """

    if not non_empty_list_of_int(labels):
        print("Error: labels must be a non-empty list of integers")
        return None

    predicted_label = labels[0]
    maximum_count = count_occurrences(predicted_label, labels)

    for label in labels:
        current_count = count_occurrences(label, labels)
        if current_count > maximum_count:
            maximum_count = current_count
            predicted_label = label

    return predicted_label


training_images = [
    [0, 50, 0,
     50, 200, 50,
     0, 50, 0],        # Image for digit 0
    [255, 255, 255,
     255, 0, 255,
     255, 255, 255]    # Image for digit 5
]

# Corresponding labels
training_labels = [0, 5]

# Example test image (still 2D)
test_image_raw = [
    [0, 50, 0],
    [50, 150, 50],
    [0, 50, 0]
]

# Flatten the test image
test_image = flatten_image(test_image_raw)

# Compute distances of test image to all training images
distances = compute_distances(test_image, training_images, training_labels)

# Set k to different values for test
k = 1
#selected for demonstration on small dataset
#is simple and straightforward
#shows clearly how the program picks the nearest neighbor and predicts

# Find k nearest neighbors for both Manhattan and Euclidean
neighbors_manhattan, neighbors_euclidean = k_nearest_neighbors(distances, k)

# Extract labels
labels_manhattan = []
for neighbor in neighbors_manhattan:
    labels_manhattan.append(neighbor[0])

labels_euclidean = []
for neighbor in neighbors_euclidean:
    labels_euclidean.append(neighbor[0])

# Predict label using majority vote for each distance metrics
predicted_manhattan = majority_vote(labels_manhattan)
predicted_euclidean = majority_vote(labels_euclidean)

# Output results
print(f"Predicted label (Manhattan): {predicted_manhattan}")
print(f"Predicted label (Euclidean): {predicted_euclidean}")
