# Simple tetris program! v0.2
# D. Crandall, Sept 2016


'''
For this problem we have implemented an AI for winning the Tetris game. The main idea behind this code is as follows:
- for a given piece, create a search tree with all the possible boards considering all the possible locations and rotations for that piece. 
  The successor function returns the new board, the goodness score of that board and the final location of the piece. 
- For each new board created in the successor function, we expand it again considereing the next piece (tetris.get_next_piece()). 
  The final position of the current piece (given by tetris.get_piece()) is determined by the new board with max goodness score at this point.
  In this sense, we are evaluating the final position of the current piece conisdering as well the information of the next piece. 
- After we extracted the final destination, the piece is placed following the simple and animated version. 
- The goodness score is determined by various factors on the board such as the number of complete lines, the number of holes, 
  the bumpiness, and the number of complete lines. This gives us an idea of how good is the current board. 


Note: We have also created a distribution table for the incoming pieces in order to learn that and try to estimate the move based on
the incoming piece we believe will come. This incoming piece is determined based on the distribution table from which we extrac the piece
with the highest probability. With this set, we expand the tree from the previous states and get the new score and final best move. 
This implementation had to be removed from the code since it takes a lot of time and he animated version is not able to run properly
because of this.  
'''




from AnimatedTetris import *
from SimpleTetris import *
from kbinput import *
import time, sys
from random import randint
import operator


class HumanPlayer:
    def get_moves(self, tetris):
        print "Type a sequence of moves using: \n  b for move left \n  m for move right \n  n for rotation\nThen press enter. E.g.: bbbnn\n"
        moves = raw_input()
        return moves

    def control_game(self, tetris):
        while 1:
            c = get_char_keyboard()
            commands = {"b": tetris.left, "n": tetris.rotate, "m": tetris.right, " ": tetris.down}
            commands[c]()


#####
# This is the part you'll want to modify!
# Replace our super simple algorithm with something better
#
class ComputerPlayer:

    all_rotations = {}
    piece_distribution = {}
    num_of_pieces = 0

    def __init__(self):
        self.build_rotation_dic()
        self.build_piece_dist_dic()

    # This function should generate a series of commands to move the piece into the "optimal"
    # position. The commands are a string of letters, where b and m represent left and right, respectively,
    # and n rotates. tetris is an object that lets you inspect the board, e.g.:
    #   - tetris.col, tetris.row have the current column and row of the upper-left corner of the 
    #     falling piece
    #   - tetris.get_piece() is the current piece, tetris.get_next_piece() is the next piece after that
    #   - tetris.left(), tetris.right(), tetris.down(), and tetris.rotate() can be called to actually
    #     issue game commands
    #   - tetris.get_board() returns the current state of the board, as a list of strings.
    #
    def get_moves(self, tetris):

        self.num_of_pieces += 1 


        piece = tetris.get_piece()[0]

        # updating distribution table
        # self.piece_distribution[tuple(piece)][0] += 1
        # for key, value in self.piece_distribution.iteritems():
        #     value[1] = float(value[0]) / self.num_of_pieces

        board = tetris.get_board()
        piece = tetris.get_piece()
        next_piece = tetris.get_next_piece()

        succ_good = successors(tetris, board, piece[0])
        

        # possible_next_piece =  max(self.piece_distribution.iteritems(), key=operator.itemgetter(1))[0]

        final_succ = []
        for b in succ_good:
            aux_succ = successors(tetris, b[-1], next_piece)
            for succ in aux_succ:
                succ = succ + [b[1], b[2]]
                final_succ += [succ]


        scores = [x[0] for x in final_succ]

        best_board_idx = scores.index(max(scores))

        # getting the final col given by the move with highest goodness score
        final_col = final_succ[best_board_idx][-2]

        # getting the rotation id given by the move with highest goodness score
        rot_idx = final_succ[best_board_idx][-1]

        piece, piece_row, piece_col = tetris.get_piece()

        return self.get_cmd(piece_col, final_col, rot_idx)



    # This is the version that's used by the animted version. This is really similar to get_moves,
    # except that it runs as a separate thread and you should access various methods and data in
    # the "tetris" object to control the movement. In particular:
    #   - tetris.col, tetris.row have the current column and row of the upper-left corner of the
    #     falling piece
    #   - tetris.get_piece() is the current piece, tetris.get_next_piece() is the next piece after that
    #   - tetris.left(), tetris.right(), tetris.down(), and tetris.rotate() can be called to actually
    #     issue game commands
    #   - tetris.get_board() returns the current state of the board, as a list of strings.
    #
    def control_game(self, tetris):

        while 1:
            time.sleep(0.1)

            self.num_of_pieces += 1 

            piece = tetris.get_piece()[0]

            # self.piece_distribution[tuple(piece)][0] += 1
            # self.piece_distribution[tuple(piece)][1] = float(self.piece_distribution[tuple(piece)][0]) / self.num_of_pieces

            board = tetris.get_board()
            piece = tetris.get_piece()
            next_piece = tetris.get_next_piece()

            succ_good = successors(tetris, board, piece[0])

            final_succ = []
            for b in succ_good:
                aux_succ = successors(tetris, b[-1], next_piece)
                # positions += [b[1], b[2]]
                for succ in aux_succ:
                    succ = succ + [b[1], b[2]]
                    final_succ += [succ]


            scores = [x[0] for x in final_succ]

            best_board_idx = scores.index(max(scores))

            # getting the final col given by the move with highest goodness score
            final_col = final_succ[best_board_idx][-2]

            # getting the rotation id given by the move with highest goodness score
            rot_idx = final_succ[best_board_idx][-1]

            for idx in xrange(0, rot_idx):
                tetris.rotate()

            if (final_col < tetris.col):
                tetris.left()
            elif (final_col > tetris.col):
                tetris.right()
            else:
                tetris.down()


    # This function builds a static dictionary of pieces and its rotations
    #
    def build_rotation_dic(self):
        
        self.all_rotations[tuple(["xx", "xx"])] = []

        self.all_rotations[tuple(["x", "x", "x", "x"])] = []
        self.all_rotations[tuple(["x", "x", "x", "x"])].append(["xxxx"])

        self.all_rotations[tuple(["xxxx"])] = []
        self.all_rotations[tuple(["xxxx"])].append(["x", "x", "x", "x"])

        self.all_rotations[tuple(["xx ", " xx"])] = []
        self.all_rotations[tuple(["xx ", " xx"])].append([" x", "xx", "x "])

        self.all_rotations[tuple([" x", "xx", "x "])] = []
        self.all_rotations[tuple([" x", "xx", "x "])].append(["xx ", " xx"])

        self.all_rotations[tuple(["xxx", "  x"])] = []
        self.all_rotations[tuple(["xxx", "  x"])].append([" x", " x", "xx"])
        self.all_rotations[tuple(["xxx", "  x"])].append(["x  ", "xxx"])
        self.all_rotations[tuple(["xxx", "  x"])].append(["xx", "x ", "x "])

        self.all_rotations[tuple([" x", " x", "xx"])] = []
        self.all_rotations[tuple([" x", " x", "xx"])].append(["x  ", "xxx"])
        self.all_rotations[tuple([" x", " x", "xx"])].append(["xx", "x ", "x "])
        self.all_rotations[tuple([" x", " x", "xx"])].append(["xxx", "  x"])

        self.all_rotations[tuple(["x  ", "xxx"])] = []
        self.all_rotations[tuple(["x  ", "xxx"])].append(["xx", "x ", "x "])
        self.all_rotations[tuple(["x  ", "xxx"])].append(["xxx", "  x"])
        self.all_rotations[tuple(["x  ", "xxx"])].append([" x", " x", "xx"])

        self.all_rotations[tuple(["xx", "x ", "x "])] = []
        self.all_rotations[tuple(["xx", "x ", "x "])].append(["xxx", "  x"])
        self.all_rotations[tuple(["xx", "x ", "x "])].append([" x", " x", "xx"])
        self.all_rotations[tuple(["xx", "x ", "x "])].append(["x  ", "xxx"])

        self.all_rotations[tuple(["xxx", " x "])] = []
        self.all_rotations[tuple(["xxx", " x "])].append([" x", "xx", " x"])
        self.all_rotations[tuple(["xxx", " x "])].append([" x ", "xxx"])
        self.all_rotations[tuple(["xxx", " x "])].append(["x ", "xx", "x "])

        self.all_rotations[tuple([" x", "xx", " x"])] = []  # T
        self.all_rotations[tuple([" x", "xx", " x"])].append([" x ", "xxx"])
        self.all_rotations[tuple([" x", "xx", " x"])].append(["x ", "xx", "x "])
        self.all_rotations[tuple([" x", "xx", " x"])].append(["xxx", " x "])

        self.all_rotations[tuple([" x ", "xxx"])] = []
        self.all_rotations[tuple([" x ", "xxx"])].append(["x ", "xx", "x "])
        self.all_rotations[tuple([" x ", "xxx"])].append(["xxx", " x "])
        self.all_rotations[tuple([" x ", "xxx"])].append([" x", "xx", " x"])

        self.all_rotations[tuple(["x ", "xx", "x "])] = []
        self.all_rotations[tuple(["x ", "xx", "x "])].append(["xxx", " x "])
        self.all_rotations[tuple(["x ", "xx", "x "])].append([" x", "xx", " x"])
        self.all_rotations[tuple(["x ", "xx", "x "])].append([" x ", "xxx"])


    def build_piece_dist_dic(self):
        
        self.piece_distribution[tuple(["xx", "xx"])] = [0, 0]
        self.piece_distribution[tuple(["x", "x", "x", "x"])] = [0, 0]
        self.piece_distribution[tuple(["xxxx"])] = [0, 0]
        self.piece_distribution[tuple(["xx ", " xx"])] = [0, 0]
        self.piece_distribution[tuple([" x", "xx", "x "])] = [0, 0]
        self.piece_distribution[tuple(["xxx", "  x"])] = [0, 0]
        self.piece_distribution[tuple([" x", " x", "xx"])] = [0, 0]
        self.piece_distribution[tuple(["x  ", "xxx"])] = [0, 0]
        self.piece_distribution[tuple(["xx", "x ", "x "])] = [0, 0]
        self.piece_distribution[tuple(["xxx", " x "])] = [0, 0]
        self.piece_distribution[tuple([" x", "xx", " x"])] = [0, 0]
        self.piece_distribution[tuple([" x ", "xxx"])] = [0, 0]
        self.piece_distribution[tuple(["x ", "xx", "x "])] = [0, 0]

    # This function returns a string with the commands needed to perform the best move possible given by our algorithm
    #
    def get_cmd(self, piece_col, final_col, rot_idx):
        
        rot = "n" * rot_idx
        disp = ''
        if piece_col > final_col:
            disp = "b" * abs(piece_col - final_col)
        if piece_col < final_col:
            disp = "m" * abs(piece_col - final_col)

        commands = rot + disp
        return commands



# This function rotates the tetris piece
# It returns a list of the piece followed by its possible rotations given by the rotation dictionary
#
def rotations(tetris, piece):
    rotation = list(ComputerPlayer.all_rotations[tuple(piece)])
    rotation.insert(0, piece)
    return rotation


# This function generates successors for the given tetris piece. 
# - Given a piece and a board state, it generates all the possible boards and it calculates its goodness score based 
#   on a heuristic function
#
def successors(tetris, board, piece):
    succ = []
    ele = 'x'
    rot_idx = 0
    base_row = len(board)

    for rotated_piece in rotations(tetris, piece):
        
        for all_locations in xrange(0, TetrisGame.BOARD_WIDTH - len(rotated_piece[0]) + 1):
        
            col = zip(*board)[all_locations]  # returns a tuple for the specified 'all_locations' column
            row_to_put = col.index(ele) if ele in col else base_row

            if not TetrisGame.check_collision((board, 0), rotated_piece, row_to_put - len(rotated_piece), all_locations):
                succ_board, score = TetrisGame.place_piece((board, 0), rotated_piece, row_to_put - len(rotated_piece), all_locations)

                succ_heu = goodness(succ_board)
                succ.append([succ_heu, all_locations, rot_idx, succ_board])
        rot_idx += 1

    return succ


# This function returns the goodness function for a given board. This goodness function consists of a linear combination of:
# - number of aggregated heights: sum of height of each column. We try to minimize this value
# - number of complete lines. This is a value we want to maximize
# - number of holes in the given board. We try to minimize this value
# - variation of board's column heights. We try to minimize this value as we want the board to be as monotone as possible
# 
# Each of these values are weighted by a experimental factors which were determined following this blog: 
# https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/ but then modified based on our experiments
# 
def goodness(board):
    agg = get_aggregate_height(board)
    lines = get_complete_lines(board)
    holes = get_holes_count(board)
    bumpiness = get_bumpiness_value(board)
    max_height = get_max_height(board)
    min_height = get_min_height(board)

    return float(agg) * -0.610066 \
           + float(lines) * 0.760666 \
           + float(holes) * -0.45663 \
            + float(bumpiness) * -0.184483


def get_aggregate_height(board):
    ele = 'x'
    base_height = 0
    column_height = 0
    row = 0

    for col in xrange(0, TetrisGame.BOARD_WIDTH):
        column = zip(*board)[col]
        column_height += TetrisGame.BOARD_HEIGHT - column.index(ele) if ele in column else base_height
    return column_height


def get_max_height(board):
    ele = 'x'
    base_height = 0
    column_height = []

    row = 0

    for col in xrange(0, TetrisGame.BOARD_WIDTH):
        column = zip(*board)[col]
        column_height.append(TetrisGame.BOARD_HEIGHT - column.index(ele) if ele in column else base_height)
    return max(column_height)

def get_min_height(board):
    column_heights = [ min([ r for r in range(len(board)-1, 0, -1) if board[r][c] == "x"  ] + [100,] ) for c in range(0, len(board[0]) ) ]
    index = column_heights.index(max(column_heights))
    return index


def get_bumpiness_value(board):
    ele = 'x'
    base_height = TetrisGame.BOARD_HEIGHT
    bumpiness = 0
    for cols in xrange(0, TetrisGame.BOARD_WIDTH - 1):
        column1 = zip(*board)[cols]
        column2 = zip(*board)[cols + 1]
        column1_height = column1.index(ele) if ele in column1 else base_height
        column2_height = column2.index(ele) if ele in column2 else base_height
        bumpiness += abs(column1_height - column2_height)
    return bumpiness


def get_holes_count(board):
    ele = 'x'
    base_height = TetrisGame.BOARD_HEIGHT
    holes_count = 0
    for cols in xrange(0, TetrisGame.BOARD_WIDTH):
        column = zip(*board)[cols]
        holes_count += column[column.index(ele) if ele in column else base_height:base_height].count(' ')
    return holes_count


def get_complete_lines(board):
    complete = [i for (i, s) in enumerate(board) if s.count(' ') == 0]
    return len(complete)


################### main program
(player_opt, interface_opt) = sys.argv[1:3]

try:
    if player_opt == "human":
        player = HumanPlayer()
    elif player_opt == "computer":
        player = ComputerPlayer()
    else:
        print "unknown player!"

    if interface_opt == "simple":
        tetris = SimpleTetris()
    elif interface_opt == "animated":
        tetris = AnimatedTetris()
    else:
        print "unknown interface!"

    tetris.start_game(player)

except EndOfGame as s:
    print "\n\n\n", s
