#!/usr/bin/python
#
# Mountain ridge finder
# Based on skeleton code by D. Crandall, Oct 2016
#
'''
Part 1 :
    The Bayes net in figure 1b signifies that we should take the maximum gradient value for each row, and the resultant list would be our ridge.
Part 2 :
    We take 10 maximum gradient value from column 0 for given image, and then we find ridge list starting all 10 rows that we found in earlier step.
    We select row for next column after row 0 and recursively other based on emission probability.
    Out of these 10 ridges we select the ridge with minimum bumpiness.
Part 3 :
    Here we are already given a point on ridge by a user, we find the ridge from this point towards star and end column respectively.
    We select row number using emission probability
'''
from PIL import Image
from numpy import *
from scipy.ndimage import filters
from scipy.misc import imsave
import sys
import heapq

# calculate "Edge strength map" of an image
#
from sympy.polys.subresultants_qq_zz import final_touches


def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale,0,filtered_y)
    return filtered_y**2

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_edge(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( max(y-thickness/2, 0), min(y+thickness/2, image.size[1]-1 ) ):
            image.putpixel((x, t), color)
    return image

# main program
#
(input_filename, output_filename, gt_row, gt_col) = sys.argv[1:]

# load in image
input_image = Image.open(input_filename)

# compute edge strength mask
edge_strength = edge_strength(input_image)
imsave('edges.jpg', edge_strength)

def part2(edge_strength):
    global dict_emission_prob
    max_points_initial = 10
    initial_col = edge_strength[:,0]
    max_points = heapq.nlargest(max_points_initial, range(len(initial_col)), initial_col.__getitem__)
    temp_max = []
    final_dict = {}

    for i in max_points:
        final_dict[i] = [0]

    for points in max_points:

        dict_ridge = [0]
        dict_ridge[0] = points

        for col in range(1, edge_strength.shape[1]):
            for row in range(0,edge_strength.shape[0]):

                prob = (abs((row - points)/edge_strength.shape[0]))*dict_emission_prob[row][col]
                temp_max.append(prob)
            max_score = -1
            for values in range(0,len(temp_max)-1):
                if max_score < float(temp_max[values]):
                    max_score = float(temp_max[values])
                    index = values
            dict_ridge.append(index)
            temp_max=[]
        final_dict[points] = dict_ridge

    score = 0
    min = sys.maxint
    min_ridge = 0

    for ridges in final_dict:
        for i in range(0, len(final_dict[ridges])-1):
            score += abs (final_dict[ridges][i] - final_dict[ridges][i+1] )
        if min > score:
           min = score
           min_ridge = ridges
        score = 0

    return final_dict[min_ridge]


def part3(edge_strength, gt_row, gt_col):
    global dict_emission_prob
    temp_max = []

    # # for points in max_points:
    dict_ridge = [0]*edge_strength.shape[1]
    dict_ridge[gt_col] = gt_row
    for col in range(gt_col-1, -1, -1):
        for row in range(0, edge_strength.shape[0]):
            prob = (abs((row - gt_row) / edge_strength.shape[0])) * dict_emission_prob[row][col]
            temp_max.append(prob)
        max_score = -1
        for values in range(0, len(temp_max) - 1):
            if max_score < float(temp_max[values]):
                max_score = float(temp_max[values])
                index = values
        dict_ridge[col] = index
        temp_max = []

    # dict_ridge[gt_col] = gt_row

    for col in range(gt_col+1, edge_strength.shape[1]):
        for row in range(0, edge_strength.shape[0]):
            prob = (abs((row - gt_row) / edge_strength.shape[0])) * dict_emission_prob[row][col]
            temp_max.append(prob)
        max_score = -1
        for values in range(0, len(temp_max) - 1):
            if max_score < float(temp_max[values]):
                max_score = float(temp_max[values])
                index = values
        dict_ridge[col] = index
        temp_max = []

    return dict_ridge


# You'll need to add code here to figure out the results! For now,
# just create a horizontal centered line.
# ridge = [ edge_strength.shape[0]/2 ] * edge_strength.shape[1]

#Part1
ridge = edge_strength.argmax(axis=0)

#Part2
global dict_emission_prob
dict_emission_prob = zeros(edge_strength.shape)
for col in range(0, edge_strength.shape[1]):
    for row in range(0, edge_strength.shape[0]):
        dict_emission_prob[row][col] = 0

max_col = edge_strength.max(axis=0)
for col in range(0, edge_strength.shape[1]):
    for row in range(0, edge_strength.shape[0]):
        dict_emission_prob[row][col] = edge_strength[row][col]/max_col[col]


ridge_basic = part2(edge_strength)
ridge_human = part3(edge_strength, int(gt_row), int(gt_col))

imsave(output_filename, draw_edge(input_image, ridge, (255, 0, 0), 5))
input_image = Image.open(output_filename)
imsave(output_filename, draw_edge(input_image, ridge_basic, (0, 0, 255), 5))
input_image = Image.open(output_filename)
imsave(output_filename, draw_edge(input_image, ridge_human, (0, 255, 0), 5))
