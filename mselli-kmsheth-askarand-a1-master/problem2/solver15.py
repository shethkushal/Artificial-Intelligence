'''
(1) a description of how you formulated the search problem, including precisely defining the state space, the
successor function, the edge weights, and (if applicable) the heuristic function(s) you designed, including an
argument for why they are admissible; 
A: 
state space: The state space would be of size 15! since there are 15 tiles that can be placed anywhere.

successor function: Generates state by sliding the tile into the empty tile. Considering the special case of the problem
an empty tile can get a tile from all the four sides.

edge weights: current cost + heuristic cost, where current cost is the cost to reach that state plus the heuristic cost
computed by the chosen heuristic function.

heuristic function: manhattan distance. It is admissible because it will never overestimate the cost of moving the tile, since
the only possible movement is vertical or horizontal.


(2) a brief description of how your search algorithm works; 
A:
Implemented the A* search algorithm, wherein it explores the best possible states of the puzzle whenever possible and reaches 
the goal state.

(3) and discussion of any problems you faced, any assumptions, simplfications, and/or design decisions you made.
A:
problems and its solutions:
1)For faster lookup, replaced the PriorityQueue with the dictionary.
2)The dictionary is a entry with puzzle: cost relation, so since the puzzle was list of list (which is not hashable), converted the list
into tuple form and then used it as the key.

'''



'''
Tested the solver15.py with two heuristics:

1) Manhattan distance
2) Misplaced tiles

The performance for the following input puzzle :
1 6 2 4
5 10 7 8
11 9 15 0
13 14 3 12

with Manhattan distance:	 0.0369880199432 s
with Misplaced tile:		 0.0729999542236 s
'''


'''

Entry point for the problem. Refer README.txx file for instructions to run this program.

'''

import os
import sys
import time
from astar import astar

endl = os.linesep


arguments = sys.argv

if arguments==None or len(arguments) <=1:
	# Less # of arguments passed to the program
	print "Insufficient arguments"
	print "Try python solver15.py <input-board-filename> [print result flag condition]"
	sys.exit(1)

filepath = arguments[1]

print_result_flag = False

'''
If the print result flag is set to something then check, otherwise go ahead with the default config
'''
if len(arguments) > 2:
	if arguments[2] <> "True" and arguments[2] <> "False":
		print "Flag value error: Accepted values are (True, False)"
		print "Try python solver15.py <input-board-filename> [print result flag condition]"
		sys.exit(1)
	else:
		if arguments[2] == "True":
			print_result_flag = True
		else:
			print_result_flag = False

'''
Read the puzzle from the file
'''
with open(filepath,"r") as f:
    initial_puzzle = [line.rstrip(endl).split(" ") for line in f.readlines()]

N = len(initial_puzzle)
'''
The input puzzle will be list (of list) of strings.
So convert it into ints.
'''
for i in xrange(0,N):
    initial_puzzle[i] = [int(initial_puzzle[i][j]) for j in xrange(0,N)]

print "Initial puzzle config : "
print "\n".join([" ".join([str(col) for col in row]) for row in initial_puzzle])


start = time.time()
print "" if astar(initial_puzzle, print_result_flag) else "Failed"

end = time.time()
print "Completed in ",end - start," s."

sys.exit(0)