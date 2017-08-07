'''
Responsible for generating the successors for the puzzle
'''

import copy
from puzzle_state import Puzzle_State
from heuristics import calculate_manhattan_distance
# Only for tesing:-
# from heuristics import calculate_misplaced_tiles

# ith tile would be empty[0][0] means the row
# jth tile would be empty[0][1] means the column

def move_up(empty, N):
    tile_i_to_move = (empty[0][0] + 1) % N
    tile_j_to_move = empty[0][1]
    return [tile_i_to_move, tile_j_to_move]

def move_down(empty, N):
    tile_i_to_move = (empty[0][0] - 1) % N
    tile_j_to_move = empty[0][1]
    return [tile_i_to_move, tile_j_to_move]

def move_left(empty, N):
    tile_i_to_move = empty[0][0]
    tile_j_to_move = (empty[0][1]  + 1) % N
    return [tile_i_to_move, tile_j_to_move]

def move_right(empty, N):
    tile_i_to_move = empty[0][0]
    tile_j_to_move = (empty[0][1] - 1) % N
    return [tile_i_to_move, tile_j_to_move]

def get_empty_tile_location(puzzle):
    N = len(puzzle)
    return [[r,c] for r in xrange(0,N) for c in xrange(0,N) if puzzle[r][c] == 0]

'''
the puzzle here is the tuple which we got from the fringe
Return the new successor tuple
'''
def successor(puzzle, move):
    empty = get_empty_tile_location(puzzle)
    N = len(puzzle)
    new_puzzle = copy.deepcopy(puzzle)

    # for converting tuple to list
    # new_puzzle = [list(ele) for ele in puzzle]
    new_puzzle = map(list, puzzle)

    moves = {'L': move_left, 'R':move_right, 'D':move_down, 'U':move_up}
    function = moves[move]
    tile_i_j_to_move = function(empty, N)

    tile_i, tile_j = tile_i_j_to_move[0:2]
    #tile_j = tile_i_j_to_move[1]

    # do the swapping ritual
    new_puzzle[tile_i][tile_j], new_puzzle[empty[0][0]][empty[0][1]] = \
        new_puzzle[empty[0][0]][empty[0][1]], new_puzzle[tile_i][tile_j]

    # converting the successor function back to tuple

    new_tuple = tuple(tuple(ele) for ele in new_puzzle)

    return new_tuple


def successors(puzzle_state):
	puzzle1, puzzle2, puzzle3, puzzle4 = [successor(puzzle_state.puzzle, move) for move in ('L', 'R', 'U', 'D')]
	man_dist1, man_dist2, man_dist3, man_dist4 = [calculate_manhattan_distance(puzzle) for puzzle in (puzzle1, puzzle2, puzzle3, puzzle4)]
    # Only for tesing:-
    # man_dist1, man_dist2, man_dist3, man_dist4 = [calculate_misplaced_tiles(puzzle, len(puzzle)) for puzzle in (puzzle1, puzzle2, puzzle3, puzzle4)]
	return [Puzzle_State(puzzle1, man_dist1, puzzle_state.current_cost + man_dist1,'L'),
	        Puzzle_State(puzzle2, man_dist2, puzzle_state.current_cost + man_dist2,'R'),
	        Puzzle_State(puzzle3, man_dist3, puzzle_state.current_cost + man_dist3,'U'),
	        Puzzle_State(puzzle4, man_dist4, puzzle_state.current_cost + man_dist4,'D' )]