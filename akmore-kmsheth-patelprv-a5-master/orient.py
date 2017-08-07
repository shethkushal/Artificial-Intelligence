'''
    K NEAREST NEIGHBOUR:

    In K nearest neighbour we find the euclidean distance between each testing and training file. We store the image from
    the train data which has the least distance from the test data. We then assign the orientation of the train file which
    is at the least distance to the test file.
    The confusion matrix for nearest neighbour is as given below:

                  |       0       |      90      |     180      |     270
    ---------------------------------------------------------------------------
         0        |      155      |      24      |      36      |      24
    ---------------------------------------------------------------------------
         90       |       19      |     154      |      19      |      32
    ---------------------------------------------------------------------------
        180       |       38      |      28      |     154      |      16
    ---------------------------------------------------------------------------
        270       |       20      |      33      |      20      |     171
    ---------------------------------------------------------------------------

    Accuracy: 67.23%

    ADABOOST:

    We select 960 random points to train the data. the x co-ordinates are selected from 1 to 192 and for each value of x
    we generate random 5 y values. There is a check to ensure that the random points are not repeated.
    To classify image in each orientation (0, 90, 180, 270)
    To train the model to classify the images in one of the 4 orientations (0, 90, 180, 270) we perform the following
    steps for each orientation classification:
    Assign a weight of [1/Number of  training images] to each image in the train file

    Iteration for 0 orientation:
    We compare the pixel values at the x and y co-ordinate for each image in the training set.

    If the value at x pixel is greater than the value at y pixel and the orientation of the train file image and the
    current orientation (i.e. 0) is the same then the image is correctly classified.
    If the value at x pixel is less than the value at pixel y, and the orientation of the train file image is not equal
    to 0 then the image is correctly classified.

    This process is done for the remaining 3 orienations (90, 180, 270) as well.

    The images which are correcly classified by the model are stored in a list.  We then update the weights of the
    rightly classified files.
    New weight = old weight * (1*beta) where beta= e/(1-e) where e= 1-score

    We then normalize the weights of all the images.
    We then calculate the confidence score (weight of each hypothesis). The

    We continue this process for the number of stumps provided by the user.

    We store the confidence score of each stump for each orientation, the x and y pixel indexes of the maximum score are stored.

    During testing we assign for each stump we check the value of x and y for each stump. If the value at x pixel is
    greater than the value at y pixel then we increase the confidence in thetest_file_confidence list for that orientation.
    The orientation which has the maximum value in the test_file_confidence list is the orientation of the test file.

    The confusion matrix for different values of decision stumps is as given below:


    Decision Stump count = 80

                  |       0       |      90      |     180      |     270
    ---------------------------------------------------------------------------
         0        |      177      |      24      |      23      |      15
    ---------------------------------------------------------------------------
         90       |       22      |     162      |      15      |      25
    ---------------------------------------------------------------------------
        180       |       41      |      30      |     144      |      21
    ---------------------------------------------------------------------------
        270       |       29      |      28      |      16      |     171
    ---------------------------------------------------------------------------

    Accuracy: 69.35%


    NEURAL NETS:

    The input nodes in the neural network are the 192 pixels of the image. The 192 input nodes are then passed to the
    hidden nodes. The hidden nodes count is the input from the user.
    The summation of the dot product of weight and pixel value is passed to the sigmoid function.
    The sigmoid function is 1/1+e^-x.
    The same procedure is repeated for hidden node as the input and the orientations (0, 90, 180, 270) as the output nodes.
    Then we calculate the error for all the orientations. The errors are then back propagated and the weights are updated
    accordingly.
    We repeat this process for 3 iterations.

    We pass the input image through the trained neural network and calculate the output at each node in the output layer.
    We pick the maximum value of the output node and take its corresponding orientation. We calculate the accuracy of
    this model by comparing the estimated orientation with actual image orientation.

    The confusion matrix are as given below:

    Hidden Node count= 12

                  |       0       |      90      |     180      |     270
    ---------------------------------------------------------------------------
         0        |      160      |      10      |      19      |      50
    ---------------------------------------------------------------------------
         90       |       12      |     154      |      5       |      53
    ---------------------------------------------------------------------------
        180       |       27      |      17      |     141      |      51
    ---------------------------------------------------------------------------
        270       |       8       |      17      |      3       |     216
    ---------------------------------------------------------------------------

    Accuracy: 71.16%


    BEST:
    We found that the accuracy varies due to random weight assigned in the first iteration of neural net. But if we take
    hidden node count greater than 5, we observe that accuracy oscillates between 70 to 72 %. We dumped a model file for
    hidden node count = 12, which is named as 'model_file'. To use the script in Best mode, pass the file as the 5th parameter
    to the script. We have skipped the training as training neural net might take some time, hence instead we use the model
    file. Using this model file we test the testing data and we get the accuracy of 71.58 %

    Confusion Matrix:

                  |       0       |      90      |     180      |     270
    ---------------------------------------------------------------------------
         0        |      157      |      30      |      22      |      30
    ---------------------------------------------------------------------------
         90       |       14      |     180      |      4       |      26
    ---------------------------------------------------------------------------
        180       |       25      |      35      |     150      |      26
    ---------------------------------------------------------------------------
        270       |       13      |      36      |      7       |     188
    ---------------------------------------------------------------------------

    Accuracy: 71.58%

'''

import sys
import os
import math
import pickle
from collections import defaultdict
import random


def read_file(file_name):
    file_dict = {}
    input_file = open(file_name, 'r')
    for line in input_file:
        line = line.split()
        file_dict[(line[0], line[1])] = [int(item) for item in line[2:]]
        # file_dict[(line[0], line[1])] = map(int, line[2: ])
    return file_dict


def nearest():
    confusion_matrix = defaultdict(dd)
    accurate_count = 0

    result_file = open('nearest_output.txt', 'w')

    for test_file in test_data:
        min_distance = float('inf')
        min_orientation = 0

        for train_file in train_data:
            distance = euclidean(test_data[test_file], train_data[train_file])
            if distance < min_distance:
                min_distance = distance
                min_orientation = train_file[1]

        if min_orientation == test_file[1]:
            accurate_count += 1

        confusion_matrix[test_file[1]][min_orientation] += 1

        result_file.write(str(test_file[0]) + " " + min_orientation + "\n")

    result_file.close()

    # Printing confusion matrix for Nearest neighbour
    print_results(confusion_matrix, accurate_count)


def adaboost():
    print "Adaboost"

    if len(sys.argv) < 5:
        print "Too few arguments, please input [train_file] [test_file] [mode] [stump_count] in argument"
        exit()

    stump_count = int(sys.argv[4])

    weights = {'0': {}, '90': {}, '180': {}, '270': {}}
    initial_weight = float(1) / float(len(train_data))

    confidence = {'0': [], '90': [], '180': [], '270': []}

    result_file = open('adaboost_output.txt', 'w')

    for train_index in train_data:
        weights['0'][train_index] = initial_weight
        weights['90'][train_index] = initial_weight
        weights['180'][train_index] = initial_weight
        weights['270'][train_index] = initial_weight

    # Count of pixels
    pixel_count = len(train_data.itervalues().next())

    # Generating pixel_count*5 random points
    random_points = []

    for x in range(0, pixel_count):
        random_y = []
        while len(random_y) < 5:
            temp = random.randint(0, pixel_count - 1)
            if x != temp and temp not in random_y:
                random_y.append(temp)

        for y in random_y:
            random_points.append((x, y))

    # Training
    for i in range(0, stump_count):
        max_score = {'0': - float('inf'), '90': - float('inf'), '180': - float('inf'), '270': - float('inf')}
        max_indices = {'0': (0, 0), '90': (0, 0), '180': (0, 0), '270': (0, 0)}
        classified_files = {'0': [], '90': [], '180': [], '270': []}
        beta = {}
        for x, y in random_points:
            current_score = {'0': 0, '90': 0, '180': 0, '270': 0}
            current_classified_files = {'0': [], '90': [], '180': [], '270': []}
            for train_file in train_data:
                if train_data[train_file][x] > train_data[train_file][y]:
                    for orientation in current_score:
                        if train_file[1] == orientation:
                            current_score[orientation] += weights[orientation][train_file]
                            current_classified_files[orientation].append(train_file)
                else:
                    for orientation in current_score:
                        if train_file[1] != orientation:
                            current_score[orientation] += weights[orientation][train_file]
                            current_classified_files[orientation].append(train_file)
            # Checking the if the current score is greater than previous max
            for orientation in current_score:
                if current_score[orientation] > max_score[orientation]:
                    max_score[orientation] = current_score[orientation]
                    max_indices[orientation] = (x, y)
                    classified_files[orientation] = current_classified_files[orientation]

        for orientation in max_score:
            beta[orientation] = float(1 - max_score[orientation]) / float(max_score[orientation])
            confidence[orientation].append(
                [math.log(1.0 / float(beta[orientation])), max_indices[orientation][0], max_indices[orientation][1]])
            for file_name in classified_files[orientation]:
                weights[orientation][file_name] = beta[orientation] * weights[orientation][file_name]

        for orientation in weights:
            total_weight = sum(weights[orientation].itervalues())
            for file_name in weights[orientation]:
                weights[orientation][file_name] = float(weights[orientation][file_name]) / float(total_weight)

    # Testing
    accurate_count = 0
    confusion_matrix = defaultdict(dd)

    for test_file in test_data:
        test_file_confidence = {'0': 0, '90': 0, '180': 0, '270': 0}
        test_file_pixels = test_data[test_file]
        for orientation in confidence:
            for stumps in confidence[orientation]:
                if test_file_pixels[stumps[1]] > test_file_pixels[stumps[2]]:
                    test_file_confidence[orientation] += stumps[0]
        detected_orientation = max(test_file_confidence, key=test_file_confidence.get)
        if detected_orientation == test_file[1]:
            accurate_count += 1
        confusion_matrix[test_file[1]][detected_orientation] += 1
        result_file.write(str(test_file[0]) + " " + detected_orientation + "\n")

    print_results(confusion_matrix, accurate_count)


def nnet():
    print "Nnet"

    if len(sys.argv) < 5:
        print "Too few arguments, please input [train_file] [test_file] [mode] [stump_count] in argument"
        exit()

    # parameters for nnet
    hidden_count = int(sys.argv[4])
    alpha = 0.5
    iteration_count = 3
    output_nodes_count = 4  # Number of orientation

    # Count of pixels
    pixel_count = len(train_data.itervalues().next())

    # Training
    print "Training..."

    # Initialize weights
    weights = {}
    for i in range(pixel_count):
        for j in range(hidden_count):
            weights[(i, j + pixel_count)] = random.randrange(-1, 1) * (0.1)

    for i in range(hidden_count):
        for j in range(pixel_count):
            weights[(i + pixel_count, j + pixel_count + hidden_count)] = random.randrange(-1, 1) * (0.1)

    for iteration in range(iteration_count):
        accurate_count = 0

        for train_file in train_data:
            target_vector = get_target_vector(train_file[1])
            activation = {}
            input_dict = {}
            delta = {}

            # Forward propagation
            for input_node in range(pixel_count):
                activation[input_node] = float(train_data[train_file][input_node]) / 255

            for unit in range(pixel_count, pixel_count + hidden_count):
                input_dict[unit] = sum_1(weights, activation, unit, 0, pixel_count) + 1
                activation[unit] = g(input_dict[unit])

            output_vector = []
            for unit in range(pixel_count + hidden_count, pixel_count + hidden_count + output_nodes_count):
                input_dict[unit] = sum_1(weights, activation, unit, pixel_count, pixel_count + hidden_count) + 1
                activation[unit] = g(input_dict[unit])
                output_vector.append(activation[unit])

            estimated_orientation_index = output_vector.index(max(output_vector))
            actual_orientation_index = target_vector.index(1.0)

            if estimated_orientation_index == actual_orientation_index:
                accurate_count += 1

            # Back propagation
            for unit in range(pixel_count + hidden_count, pixel_count + hidden_count + output_nodes_count):
                delta[unit] = g_delta(input_dict[unit]) * (
                    target_vector[unit - pixel_count - hidden_count] - activation[unit])

            for unit in range(pixel_count, pixel_count + hidden_count):
                delta[unit] = g_delta(input_dict[unit]) * sum_2(weights, delta, unit, pixel_count + hidden_count,
                                                                pixel_count + hidden_count + output_nodes_count)

            # Update the weights
            for k in range(pixel_count):
                for l in range(hidden_count):
                    weights[(k, l + pixel_count)] += activation[k] * delta[l + pixel_count] * alpha

            for k in range(hidden_count):
                for l in range(output_nodes_count):
                    weights[(k + pixel_count, l + pixel_count + hidden_count)] += activation[k + pixel_count] * \
                                                                                  delta[
                                                                                      l + pixel_count + hidden_count] * alpha

    # pickle.dump(weights, open("model_file", "wb"))

    # Testing phase
    nnet_test(hidden_count, weights)


def nnet_test(hidden_count, weights):
    print "Testing..."
    output_nodes_count = 4
    accurate_count = 0
    # Count of pixels
    pixel_count = len(train_data.itervalues().next())
    confusion_matrix = defaultdict(dd)
    activation = {}
    orientations = ['0', '90', '180', '270']
    result_file = open("nnet_output.txt", "w")

    for test_file in test_data:
        target_vector = get_target_vector(test_file[1])
        input_dict = {}

        for input_node in range(pixel_count):
            activation[input_node] = float(test_data[test_file][input_node]) / 255.0

        for node in range(pixel_count, pixel_count + hidden_count):
            input_dict[node] = sum_1(weights, activation, node, 0, pixel_count) + 1
            activation[node] = g(input_dict[node])

        output_vector = []
        for node in range(pixel_count + hidden_count, pixel_count + hidden_count + output_nodes_count):
            input_dict[node] = sum_1(weights, activation, node, pixel_count, pixel_count + hidden_count) + 1
            activation[node] = g(input_dict[node])
            output_vector.append(activation[node])

        estimated_orientation_index = output_vector.index(max(output_vector))
        actual_orientation_index = target_vector.index(1.0)
        confusion_matrix[orientations[actual_orientation_index]][orientations[estimated_orientation_index]] += 1
        result_file.write(str(test_file[0]) + " " + orientations[estimated_orientation_index] + "\n")
        if estimated_orientation_index == actual_orientation_index:
            accurate_count += 1
    result_file.close()
    print_results(confusion_matrix, accurate_count)


def best():
    print "best model"
    # The neural network works best for the problem, with hidden node count = 15, I have uploaded the model_file trained
    # with neural net for that node count on github, that file is required in best mode to avoid train time
    hidden_count = 12
    if len(sys.argv) < 5:
        print "Too few arguments, please input [train_file] [test_file] [mode] [stump_count] in argument"
        exit()

    model_file_name = sys.argv[4]

    if not os.path.isfile(model_file_name):
        print "Invalid model file, exiting..."
        exit()

    weights = pickle.load(open(model_file_name, "rb"))
    nnet_test(hidden_count, weights)


def euclidean(test_image, train_image):
    # Both test and train image are of same size 192
    total = 0
    for i in range(0, len(test_image)):
        # total += math.pow((int(test_image[i]) - train_image[i]), 2)
        total += (int(test_image[i]) - train_image[i]) ** 2
    return total


def print_results(confusion_matrix, accurate_count):
    orientations = ['0', '90', '180', '270']
    print " ".center(12, ' ') + "  |  " + " | ".join(str(label).center(12, ' ') for label in orientations)
    print "".center(75, "-")
    for label_row in orientations:
        print str(label_row).center(12, ' ') + "  |  " + " | ".join(
            str(confusion_matrix[label_row][label_col]).center(12, ' ') for label_col in orientations)
        print "".center(75, "-")

    print "\n"
    print "Accuracy: " + "{0:.2%}".format(float(accurate_count) / len(test_data))


def get_target_vector(orientation):
    if orientation == '0':
        return [1.0, 0, 0, 0]
    if orientation == '90':
        return [0, 1.0, 0, 0]
    if orientation == '180':
        return [0, 0, 1.0, 0]
    if orientation == '270':
        return [0, 0, 0, 1.0]
    else:
        print "Invalid orientation in training file"
        exit()


def sum_1(weights, input_dict, unit, min, max):
    total = 0
    for i in range(min, max):
        total += weights[(i, unit)] * input_dict[i]
    return total


def sum_2(weights, input_dict, unit, min, max):
    total = 0
    for i in range(min, max):
        total += weights[(unit, i)] * input_dict[i]
    return total


# Sigmoid function
def g(x):
    return 1.0 / (1.0 + math.exp(-x))


# Derivative of the sigmoid function
def g_delta(x):
    return g(x) * (1.0 - g(x))


def dd():
    return defaultdict(int)


if __name__ == "__main__":

    if len(sys.argv) < 4:
        print "Too few arguments, please input [train_file] [test_file] [mode] [optional] in argument"
        exit()

    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    algorithm = sys.argv[3]

    algorithms = {
        "nearest": nearest,
        "adaboost": adaboost,
        "nnet": nnet,
        "best": best
    }

    if algorithm not in algorithms:
        print "Invalid algorithm, exiting..."
        exit()

    if not os.path.isfile(train_file_path):
        print "Invalid train file, exiting..."
        exit()

    if not os.path.isfile(test_file_path):
        print "Invalid test file, exiting..."
        exit()

    # Read train data
    train_data = read_file(train_file_path)

    # Read test data
    test_data = read_file(test_file_path)

    algorithms[algorithm]()
