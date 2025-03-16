"""

a dictionary game board(we defined in GUI) is converted to a numpy array format(looks similar to a list, but it works only for numerical values with C)
the changed internal data structure is like that:
new_board = [[black marbles],[white marbles]]
ex. "A1" = [1,1], "A2" = [1 ,2], "I5" = [9,5] new_board = [[[1,1], [1,2], ...],[[9,5], [9,6], ...]] 


Like what original files did, in the generate_all_next_moves, a current board is input and the program makes all the combinations of 1,2, 3 marbles,
ex. (("A1"), ("A2")) = [[1,1],], [[1,2],]   ("A1", "B2") = [[1,1],], [[2,2],]


Then, each combination is checked with move_validation, it makes all next possible boards by generate_all_next_moves.
there is all logic to check to validate marble movements(move.py). if it returns true, the move movement from source coordinate to destination coordinate is possible.

the return is like this:
result = [[[black marbles],[white marbles]], -- board 1
          [[black marbles],[white marbles]], -- board 2
          ...                                -- board n
          ]

from this point, this result can be sent to a module heuristic functions to take advantage of calculation of linear format(numpy and numba)
since heuristic function's form is like this, h(x) = w1f1 + w2f2 + w3f3 + w4f4 + ... +wnfn,
we can make it to this linear algebra format, [w1,w2,w3,w4,w5,...,wn][[f1],[f2],[f3],[f4],[f5],...[fn]] -> [1*n][n*1] = 1*1
There is nothing better than numpy dot product for calculation of equations
also, all heuristic functions need quantified vales of boards. we don't have to use change letters to values.



if you want to get the information of the next boards,
result = generate_all_next_moves(any_board, player_color)
you can use save_board_states_to_file(result, "./output/team3_test.board", player_color), there is some test codes in next_move_generator.py

if you want to apply the all boards to heuristic functions,
result = generate_all_next_moves(any_board, player_color)
you can import this result from a module of heuristic_functions_and_search

"""
