import time
import  numpy as np
import sys
global suggest_move_to
global k
global start
global available

def move(grid, N, K, move_num, starttime, available_time):
    global n
    n = N
    global k
    k = K
    global MIN
    global start
    global available
    start  = starttime
    available = available_time
    MIN = move_num
    global MAX
    MAX = -move_num
    global suggest_move_to
    suggest_move_to = move_num
    best_state = None
    best_move = None
    alpha = -sys.maxsize
    beta = sys.maxsize

    for cell_val in xrange(0,9):
        v = terminal(grid, cell_val, k)
        if v <> None:
            return v

    for _s in successors(grid, move_num):
        temp_beta = min(beta, max_value(_s[0], alpha, beta, _s[1], -move_num, start, available))
        if beta > temp_beta:
        # note the state and update the alpha
            best_state = _s[0]
            best_move = _s[1]
            beta = temp_beta
    print "Next move with", "white" if move_num == 1 else "black" , "marble"
    print "I recommend you to place at row ", best_move/N  , " column ", best_move%N
    return best_state

def diag_check(grid, row, col):
    k_count = 1
    curr_marble = grid[row][col]
    if curr_marble == 0:
        return False
    '''
    diagonal scanned - \
    '''
    r = row - 1
    c = col - 1
    while r >= 0 and c >= 0 and grid[r][c] == curr_marble:
        k_count += 1
        r -= 1
        c -= 1
    if k_count >= k:
        return True
    r = row + 1
    c = col + 1
    while r < n and c < n and grid[r][c] == curr_marble:
        k_count += 1
        r += 1
        c += 1

    if k_count >= k:
        return True

    '''
    diagonal scanned = /
    '''
    k_count = 1
    r = row + 1
    c = col - 1
    while r < n and c >= 0 and grid[r][c] == curr_marble:
        k_count += 1
        r += 1
        c -= 1
    if k_count >= k:
        return True
    r = row - 1
    c = col + 1
    while r >= 0 and c < n and grid[r][c] == curr_marble:
        k_count += 1
        r -= 1
        c += 1
    if k_count >= k:
        return True
    return False

def row_check(grid, row, col):
    curr_marble = grid[row][col]
    if curr_marble == 0:
        return False
    k_count = 1

    c = col-1
    while (c >=0 and grid[row][c] == curr_marble):
        k_count += 1
        c-=1

    if k_count >= k:
        return True

    c = col+1
    while(c < n and grid[row][c] == curr_marble):
        k_count += 1
        c+=1

    if k_count >= k:
        return True
    return False

def col_check(grid, row, col):
    curr_marble = grid[row][col]
    if curr_marble == 0:
        return False
    k_count = 1

    r = row-1
    while (r >= 0 and grid[r][col] == curr_marble):
        k_count+=1
        r-=1

    if k_count >= k:
        return True

    r = row+1
    while (r<n and grid[r][col] == curr_marble):
        k_count+=1
        r+=1

    if k_count >= k:
        return True

    return False

def terminal(grid, cell, depth):
    '''
    This function will check if the grid ia terminal state.
    This will check who has won out of MAX or MIN or it's a draw.
    :param grid:
    :return:
    '''
    # cell = 0
    # n = 3

    r = cell / n
    c = cell % n
    if grid[r][c] == 0:
        return None
    #check if the board is full
    full = True
    for i in xrange(0,n):
        if 0 in grid[i]:
            full = False
            break


    if row_check(grid, r, c) or col_check(grid, r, c) or diag_check(grid, r, c):
        return grid[r][c]
        # return 1        # whoever wins, return 1

    if full:
        return 0
    return None

def max_value(grid, alpha, beta, cell, move_num, start, available):
    '''

    :param grid:
    :param alpha:
    :param beta:
    :return:
    '''
    ret_val = terminal(grid, cell, k)
    if ret_val <> None:
        if ret_val == 0:
            return 0
        if suggest_move_to == ret_val:
            return 1
        elif suggest_move_to <> ret_val:
            return -1
    elif time.time() - start > available:
        return evaluate_grid(grid, ret_val, k)


    for _s in successors(grid, move_num):     # 1 since, the successors would be min nodes
        alpha = max(alpha, min_value(_s[0], alpha, beta, _s[1], -move_num, start, available))
        if alpha >= beta:
            return alpha
    return alpha

def min_value(grid, alpha, beta, cell, move_num, start, available):
    '''
    :param grid:
    :param alpha:
    :param beta:
    :return:
    '''
    ret_val = terminal(grid, cell, k)
    if ret_val <> None:
        if ret_val == 0:
            return 0
        if suggest_move_to == ret_val:
            return 1
        elif suggest_move_to <> ret_val:
            return -1
    elif time.time() - start > available:
        return evaluate_grid(grid, ret_val, k)

    if ret_val<>None:
        return evaluate_grid(grid, ret_val, k)
    for _s in successors(grid, move_num):         # -1 since, the succesors would be max nodes
        beta = min(beta, max_value(_s[0], alpha, beta, _s[1], -move_num, start, available))
        if alpha >= beta:
            return beta
    return beta

def add_marble(grid, r, c, turn_marble):
    return grid[0:r] + [grid[r][0:c] + [turn_marble, ] + grid[r][c + 1:]] + grid[r + 1:]

def successors(grid, turn_marble):
    '''
    This function returns the successors of the grid.
    :param grid:
    :param turn_marble: if 1 then the successors are generated for white marble (MAX player) else for black marble
    :return: list of grids
    '''
    n = len(grid)
    return [[add_marble(grid, r, c, turn_marble),r*n+c] for r in xrange(0, len(grid)) for c in xrange(0, len(grid[0])) if grid[r][c]==0 ]


def get_diagonals(grid):
    '''
    This function uses the numpy library to extract all the diagonals from the grid
    Courtsey: http://stackoverflow.com/questions/6313308/get-all-the-diagonals-in-a-matrix-list-of-lists-in-python

    :param grid:
    :return:
    '''
    # convert the grid into matrix
    grid_mat = np.array(grid)
    grid_diags = [grid_mat[::-1, :].diagonal(i) for i in range(-grid_mat.shape[0] + 1, grid_mat.shape[1])]
    grid_diags.extend(grid_mat.diagonal(i) for i in range(grid_mat.shape[1] - 1, -grid_mat.shape[0], -1))
    return [grid_d.tolist() for grid_d in grid_diags]

def eval_lists(grid_list, k):
    w = 0
    b = 0

    # Test
    cnt = grid_list.count(0)
    if cnt >= k:
        return [1,1, True]
    elif cnt > 0:
        seed = grid_list.index(0) if 0 in grid_list else 0
        w_k_stretch = 1
        b_k_stretch = 1
        s = seed
        while s > 0 and w_k_stretch <= k and (grid_list[s - 1] == 1 or grid_list[s - 1] == 0):
            w_k_stretch += 1
            s -= 1
        s = seed
        while s < len(grid_list)-1 and w_k_stretch < k and (grid_list[s + 1] == 1 or grid_list[s + 1] == 0):
            w_k_stretch += 1
            s += 1

        s = seed
        while s > 0 and b_k_stretch <= k and (grid_list[s - 1] == -1 or grid_list[s - 1] == 0):
            b_k_stretch += 1
            s -= 1
        s = seed
        while s < len(grid_list)-1 and b_k_stretch < k and (grid_list[s + 1] == -1 or grid_list[s + 1] == 0):
            b_k_stretch += 1
            s += 1
        if w_k_stretch >= k:
            w = 1
        if b_k_stretch >= k:
            b = 1
        return [w,b, False]
    return [0,0,False]


def eval_state(grid, k):
    w_b_d = [0]*3
    # first evaluate rows:
    for r in xrange(0, len(grid)):
        white,black,k_complete = eval_lists(grid[r], k)
        w_b_d[0] += white
        w_b_d[1] += black
    for c in xrange(0, len(grid[0])):
        grid_column = zip(*grid)[c]
        white, black, k_complete = eval_lists(grid_column, k)
        w_b_d[0] += white
        w_b_d[1] += black

    for d in get_diagonals(grid):
        white, black, k_complete = eval_lists(d, k)
        w_b_d[0] += white
        w_b_d[1] += black

    return w_b_d

def evaluate_grid(grid, ret_val, k):
    w_b_d = eval_state(grid, k)
    if suggest_move_to == 1:
        return w_b_d[0] - w_b_d[1]
    return w_b_d[1] - w_b_d[0]