"""
Author              : Kushal Sheth
Problem Definition  : Arrange n number of people on least number of tables such that none of the person knows the other person on the table
Variables:
    capacity      = maximum number of people that can seat on one table
    friends_list  = list of friends
Heuristics          : Pick the person with maximum friends first and assign him the table. Arrange other people likewise
Description         : In my program, firstly the program will design a bidirectional dictionary 'dict{}' that will map all the friends. 
                      Next it will generate a list of people 'people[]', where list is arranged in descending order, on basis of number of friends         
                      each person have. Person with maximum friends will come at the top
                      In next part, program will assign table to the first person from 'people[]'. Then it will pick up next person and will check 
                      that if he/she is eligible to sit on the table or it will make a new table. The tables here will be the dictionary of
                      dictionaries resulting into final table dictionary, 'dictionary'
Issues              : One approach was to try a brute force approach where all the combinations of seating would be tried and the most optimum 
                      will be returned. However it will take much more time. For example, if there are 10 people, it will take a long time
"""

import time
import sys

try:
    friends_list = sys.argv[1]
    capacity = int(sys.argv[2])
except IndexError:
        print "\n*********Improper Arguments***********\n"
        print "Enter in format python wedding.py [friends-file] [seats-per-table]"
        sys.exit(1)


myFile = open(friends_list, 'r')

dict = {}
dict2 = {}
unique = []
# myFile = open('myFriends2.txt', 'r')

# Parsing the of friends and converting to dictionary
start = time.time()
for row in myFile:
    line = row.rstrip('\r\n')
    my_list = line.split(' ')
    dict[my_list[0]] =  my_list[1: ]

    for item in my_list:
        if item not in unique:
            unique.append(item)

for item in unique:
    if item not in dict.keys():
        dict2[item] = []

for x in dict.values():
    for key in dict2:
        if key in x:
            dict2[key] += [dict.keys()[dict.values().index(x)]]
dict.update(dict2)
# print "Simple Dictionary: ", dict

# Updating the dictionary to a bidirectional dictionary
for key in dict:
   for x in dict.values():
       if key in x:
           if dict.keys()[dict.values().index(x)] not in dict[key]:
                dict[key] += [dict.keys()[dict.values().index(x)]]
# print "Bidirectional Dictionary:", dict

# Arranging a list of people such that the person with maximum friends comes on the top and person with least friends comes at the end of list
temp_dict = {}
temp_dict = dict.copy()
max = 0
people = []
person = 'null'

for j in range(0, len(temp_dict)):
    for i in range(0, len(temp_dict)):
        if len(temp_dict.values()[i]) > max:
            max = len(temp_dict.values()[i])
            person = temp_dict.keys()[i]
    people.append(person)
    temp_dict.pop(person)
    max = 0
    person = 'null'
# print "Sorted List: ",people

# Finding the optimal number of table that can accommodate all the people and appending them in a dictionary.
# The length of dictionary is the number of tables made
# Each dictionary has another dictionary inside it that shows the people seated on particular table
dictionary = {}
dictionary[0] = {}
dict3 = {}
values = []

for keys in people:
    # print keys
    for i in range(0, len(dictionary)):
        flag = 0
        for j in range(0, len(dictionary[i].values())):
            values += dictionary[i].values()[j]

        if keys not in values and len(dictionary[i]) < capacity:
            dict3[keys] = dict.__getitem__(keys)
            dictionary[i].update(dict3)
            flag = 1
            dict3 = {}
            break
        values = []
        print "Finding the optimal seating arrangement...\n"

    if(flag == 0):
        length = len(dictionary) - 1
        dict3[keys] = dict.__getitem__(keys)
        dictionary[length + 1] = {}
        dictionary[length + 1].update(dict3)
        dict3 = {}
    values = []
    print "Finding the optimal seating arrangement...\n"
# print dictionary

# Printing in a machine readable form'
print len(dictionary),
for i in range(0, len(dictionary)):
    for item in dictionary[i].keys():
        if item == dictionary[i].keys()[-1]:
            print item,
        else:
            print item +',',
# print "Time:" , time.time() - start
