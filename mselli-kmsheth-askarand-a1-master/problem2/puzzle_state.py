'''
class which is responsible for maintaining Puzzle state
'''

class Puzzle_State:
    def __init__(self, puzzle, heuristic_value, current_cost, move):
        if not isinstance(puzzle, tuple):
            self.puzzle = tuple(tuple(ele) for ele in puzzle)
        else:
            self.puzzle = puzzle
        self.heuristic_value = heuristic_value
        self.current_cost = current_cost
        self.move = move    # this takes values from 'L','R','U','D'