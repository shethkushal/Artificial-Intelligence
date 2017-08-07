'''
Contains the manhattan distance heuristic
'''
manhattan_dist = 0
def calculate_manhattan_distance(curr_puzzle):
    global manhattan_dist
    N = len(curr_puzzle)
    # reset manhattan distance
    manhattan_dist = 0
    for i in xrange(0, N):
        for j in xrange(0, N):
            if curr_puzzle[i][j] <> 0 and (i * N) + j + 1 <> curr_puzzle[i][j]:
                add_dist(curr_puzzle[i][j], i, j, N)
    return manhattan_dist

def add_dist(val, curr_i, curr_j, N):
    global manhattan_dist
    # row wise ith index
    goal_i = (val-1) / N
    # column wise jth index
    goal_j = (val-1) % N
    
    # As we are allowing the tiles in the puzzle to be wrapper around
    # we test from where the manhattan distance is minimum
    if abs(goal_i - curr_i) == (N-1):
        # this indicates the tile is on the vertical edge
        # and hence the distance should be wrapped around
        manhattan_dist += 1
    else:
        manhattan_dist += abs(goal_i - curr_i)
    if abs(goal_j - curr_j) == (N - 1):
        # this indicates the tile is on the horizontal edge
        # and hence the distance should be wrapped around
        manhattan_dist += 1
    else:
        manhattan_dist += abs(goal_j - curr_j)
    return manhattan_dist


def calculate_misplaced_tiles(curr_puzzle, N):
    global misplaced_tiles
    # reset misplaced tiles
    misplaced_tiles = 0

    for i in xrange(0, N):
        for j in xrange(0, N):
            if curr_puzzle[i][j] <> 0 and (i * N) + j + 1 <> curr_puzzle[i][j]:
                misplaced_tiles += 1
    return misplaced_tiles