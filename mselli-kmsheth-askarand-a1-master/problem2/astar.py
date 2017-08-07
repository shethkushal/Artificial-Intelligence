'''
A* algorithm implementation to solve the 15 puzzle problem
'''


from puzzle_state import Puzzle_State
from heuristics import calculate_manhattan_distance
from successors import successors
# Only for tesing:-
# from heuristics import calculate_misplaced_tiles

puzzle_fringe_cost = {}
puzzle_fringe_heur = {}
puzzle_fringe_closed = {}
puzzle_moves = {}


def print_puzzle(puzzle):
    return "\n".join([" ".join([str(col) for col in row]) for row in puzzle])


def goal_state(puzzle_state):
    return True if puzzle_state.heuristic_value <> None and puzzle_state.heuristic_value == 0 else False


def astar(main_puzzle, print_result_puzzle):
    puzzle_state = Puzzle_State(main_puzzle, calculate_manhattan_distance(main_puzzle), 0, " ")
    # Only for tesing:-
    # puzzle_state = Puzzle_State(main_puzzle, calculate_misplaced_tiles(main_puzzle, len(main_puzzle)), 0, " ")
    goal = astar_impl(puzzle_state)

    if not goal:
    	return False
    else:
        print "These are the moves you need to do : "
        if print_result_puzzle:
        	print print_puzzle(goal.puzzle)

        print goal.move
        return True


def astar_impl(puzzle_state):
    # test if the current state is already the goal state
    if goal_state(puzzle_state):
        return puzzle_state
    # fringe.put(puzzle_state(curr_puzzle, calculate_manhattan_distance(curr_puzzle, 3), None))
    # fringe.put(puzzle_state)
    '''
    instead of using the fringe as a priority queue, I will use a dictionary which will store the
    puzzle as the key and the heuristics and the current cost will act as its values.
    '''
    puzzle_fringe_cost[puzzle_state.puzzle] = puzzle_state.current_cost
    puzzle_fringe_heur[puzzle_state.puzzle] = puzzle_state.heuristic_value
    puzzle_moves[puzzle_state.puzzle] = puzzle_state.move

    while True:

        if puzzle_fringe_cost.__len__() == 0:
            return None

        curr_state = min(puzzle_fringe_cost, key=puzzle_fringe_cost.get)
        # curr_state has the puzzle which I need to delete from both the dictionaries

        popped_state = Puzzle_State(curr_state, puzzle_fringe_heur.get(curr_state), puzzle_fringe_cost.get(curr_state),
                                    puzzle_moves.get(curr_state))

        puzzle_fringe_cost.__delitem__(curr_state)
        puzzle_fringe_heur.__delitem__(curr_state)
        puzzle_moves.__delitem__(curr_state)
        puzzle_fringe_closed[curr_state] = '0'

        if goal_state(popped_state):
            return popped_state

        for state in successors(popped_state):
            # state is the successor state which I will look up in the closed fringe
            if puzzle_fringe_closed.get(state.puzzle) <> None:
                # This means that this state has already been visited.
                # So continue
                continue

            # if this state is present in the current fringe then check its cost and replace if cost is minimun

            cost = puzzle_fringe_cost.get(state.puzzle)  # this will get me the cost of the puzzle in the fringe
            if cost <> None:
                if cost > state.current_cost:
                    # this ensures that same state was present in the fringe
                    # and now its safe to compare the cost.
                    # if the condition holds true, we can just go ahead and update the cost
                    puzzle_fringe_cost[state.puzzle] = state.current_cost

                    # remember we have a different fringe for handling the heuristic values
                    # so its our responsibiity to update that one as well
                    puzzle_fringe_heur[state.puzzle] = state.heuristic_value

                    # along with this, we are also maintaing the moves that were made
                    # thus update the dictionary with moves
                    puzzle_moves[state.puzzle] = (popped_state.move + " " + state.move).strip()

            else:
                # this indicates that element was not present in the fringe thus
                # its safe to blindly insert in the fringe

                puzzle_fringe_cost[state.puzzle] = state.current_cost
                puzzle_fringe_heur[state.puzzle] = state.heuristic_value
                puzzle_moves[state.puzzle] = (popped_state.move + " " + state.move).strip()




