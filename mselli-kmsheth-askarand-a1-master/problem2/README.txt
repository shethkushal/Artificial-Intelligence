Steps to run the program:

On the command prompt or the shell terminal run the following command:
	python solver15.py <input-board-filename> [print result flag condition]

		* [print result flag condition] takes values from True and False.

SIDE NOTE:
The problem given in the assignment is taking too long to solve, so for testing purposes
I have given a flag which will print the result of the goal puzzle. Thus for printing the
result puzzle the value for print result flag condition should be set to True.

Thus the call should be 
	python solver15.py input_puzzle.txt True

If the value is not mentioned, the default behavior would take care to run the program and 
solve the puzzle.