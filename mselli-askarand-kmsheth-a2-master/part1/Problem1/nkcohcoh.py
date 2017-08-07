'''
Formulation:
This includes implementation of alpha-beta-min-max algorithm with an appropriate evaluation function.
The evaluation function will calculate the possibility of each player to win and will return the next favourable state.
The Tree will generate the successor states till the given time
The time is controlled in such a way that the tree will get expanded till the given time, and will return the best possible state evaluated by
the function within that time limit.

Search algorithm:
We used min-max algorithm with alpha beta pruning. Here the first player will always be min instead of max. This is because it
will return the min value to the root and find the player that makes a line of 'k' consecutive marbles and looses the game.
The search algorithm will always go till the given time.
Every time, it will store the best state and will return the most favourable move evaluated by the evaluation function.

Problems Faced:
One of the major problem was regarding which player will be playing next. So here we have assumed that the game always initiates with 'w' player.
Also if both the player played even number of times, the next turn will be taken by white.
One of the design decision includes the simplification that the root will be 'min' node.
'''

import sys
import time

global N
global K
from minimax import move


def get_next_best_move(init_grid, N, K, available_time):
    grid = [[0, ] * N for x in xrange(N)]
    init_grid = init_grid.replace('w', '1').replace('b', '0')
    '''
    load the grid
    '''
    for i in xrange(0, N):
        for j in xrange(0, N):

            if init_grid[i * N + j] == '1':
                grid[i][j] = 1
            elif init_grid[i * N + j] == '0':
                grid[i][j] = -1
            else:
                grid[i][j] = 0
    total = 0
    for l in grid:
        total += sum(l)
    if total == 0:
        move_num = 1
    else:
        move_num = -1

    starttime = time.time()
    best_move = move(grid, N, K, move_num, starttime, available_time)

    # It shows how the game progressed and which player wins
    if isinstance(best_move, int):
        if best_move == 1:
            print "Player with black marble wins!"
        elif best_move == -1:
            print "Player with white marble wins!"
        else:
            print "Game ends in a draw."
        print "Resultant board remains the same."
        print init_grid.replace("1", "w").replace("0", "b")
    else:
        result_grid = ""
        for i in xrange(0, N):
            for j in xrange(0, N):
                if best_move[i][j] == 1:
                    result_grid += 'w'
                elif best_move[i][j] == -1:
                    result_grid += 'b'
                else:
                    result_grid += '.'
        print result_grid


def main():
    arguments = sys.argv
    if len(arguments) < 5:
        print "Insufficient arguments"
        sys.exit(1)

    global N
    global K
    N = int(arguments[1])
    K = int(arguments[2])
    init_grid = arguments[3]
    timer = int(arguments[4])


    init_grid = init_grid.replace('w', '1').replace('b', '0')
    print init_grid
    estimated_time = 0.2  # derived from multiple runs for evaluating the grid
    get_next_best_move(init_grid, N, K, timer - estimated_time)


if __name__ == "__main__":
    main()
