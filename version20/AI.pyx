# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=True
# cython: nonecheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: infer_types=True
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp

from cython.parallel import prange, parallel, threadid
cimport openmp
import time
import cython
from collections import OrderedDict
from libc.math cimport sqrt, exp
from libc.stdlib cimport malloc, free, rand, RAND_MAX, srand, calloc
from libc.string cimport memcpy, memset
from libc.time cimport time as c_time

import numpy as np
cimport numpy as np


ctypedef np.int64_t INT64_t
ctypedef np.float64_t FLOAT64_t
ctypedef unsigned long long uint64_t

cdef struct EvaluationWeights:
    double marble_diff
    double centrality
    double push_ability
    double formation
    double connectivity

cdef EvaluationWeights DEFAULT_WEIGHTS

cdef void init_default_weights():
    global DEFAULT_WEIGHTS

    DEFAULT_WEIGHTS.marble_diff = 1.00
    DEFAULT_WEIGHTS.centrality = 0.311
    DEFAULT_WEIGHTS.push_ability = 0.622
    DEFAULT_WEIGHTS.formation = 0.617
    DEFAULT_WEIGHTS.connectivity = 1.131


def set_evaluation_weights(dict weights_dict):
    
    global DEFAULT_WEIGHTS

    if "marble_diff" in weights_dict:
        DEFAULT_WEIGHTS.marble_diff = weights_dict["marble_diff"]
    if "centrality" in weights_dict:
        DEFAULT_WEIGHTS.centrality = weights_dict["centrality"]
    if "push_ability" in weights_dict:
        DEFAULT_WEIGHTS.push_ability = weights_dict["push_ability"]
    if "formation" in weights_dict:
        DEFAULT_WEIGHTS.formation = weights_dict["formation"]
    if "connectivity" in weights_dict:
        DEFAULT_WEIGHTS.connectivity = weights_dict["connectivity"]

    return get_evaluation_weights()

def get_evaluation_weights():
    
    return {
        "marble_diff": DEFAULT_WEIGHTS.marble_diff,
        "centrality": DEFAULT_WEIGHTS.centrality,
        "push_ability": DEFAULT_WEIGHTS.push_ability,
        "formation": DEFAULT_WEIGHTS.formation,
        "connectivity": DEFAULT_WEIGHTS.connectivity
    }

cdef int MAX_MARBLES = 14
cdef int MAX_GROUPS = 100
cdef int MAX_BOARD_SIZE = 10
cdef int VALID_COORDS_COUNT = 61
cdef int DIRECTIONS_COUNT = 6

cdef int TT_SIZE = 1 << 22
cdef int TT_MASK = TT_SIZE - 1

cdef int TT_EXACT = 0
cdef int TT_LOWER = 1
cdef int TT_UPPER = 2

cdef struct TTEntry:
    uint64_t hash_key
    int depth
    double score
    int flag
    uint64_t best_move_hash
    int age

cdef TTEntry* tt_table = NULL

cdef struct CMarble:
    int row
    int col

cdef struct CGroup:
    CMarble marbles[3]
    int size

cdef int DIRECTIONS[6][2]

cdef void init_directions():
    DIRECTIONS[0][0] = 1; DIRECTIONS[0][1] = 0   # Upper right
    DIRECTIONS[1][0] = 1; DIRECTIONS[1][1] = 1   # Upper right diagonal
    DIRECTIONS[2][0] = 0; DIRECTIONS[2][1] = -1  # Left
    DIRECTIONS[3][0] = 0; DIRECTIONS[3][1] = 1   # Right
    DIRECTIONS[4][0] = -1; DIRECTIONS[4][1] = -1 # Lower left diagonal
    DIRECTIONS[5][0] = -1; DIRECTIONS[5][1] = 0  # Lower left

DIRECTIONS_NP = np.array([
    [1, 0],   # Upper right
    [1, 1],   # Upper right diagonal
    [0, -1],  # Left
    [0, 1],   # Right
    [-1, -1], # Lower left diagonal
    [-1, 0]   # Lower left
], dtype=np.int64)

cdef INT64_t[:, :] DIRECTIONS_VIEW = DIRECTIONS_NP

cdef int VALID_COORDS[61][2]

cdef void init_valid_coords():
    cdef int idx = 0

    VALID_COORDS[idx][0] = 9; VALID_COORDS[idx][1] = 5; idx += 1
    VALID_COORDS[idx][0] = 9; VALID_COORDS[idx][1] = 6; idx += 1
    VALID_COORDS[idx][0] = 9; VALID_COORDS[idx][1] = 7; idx += 1
    VALID_COORDS[idx][0] = 9; VALID_COORDS[idx][1] = 8; idx += 1
    VALID_COORDS[idx][0] = 9; VALID_COORDS[idx][1] = 9; idx += 1

    VALID_COORDS[idx][0] = 8; VALID_COORDS[idx][1] = 4; idx += 1
    VALID_COORDS[idx][0] = 8; VALID_COORDS[idx][1] = 5; idx += 1
    VALID_COORDS[idx][0] = 8; VALID_COORDS[idx][1] = 6; idx += 1
    VALID_COORDS[idx][0] = 8; VALID_COORDS[idx][1] = 7; idx += 1
    VALID_COORDS[idx][0] = 8; VALID_COORDS[idx][1] = 8; idx += 1
    VALID_COORDS[idx][0] = 8; VALID_COORDS[idx][1] = 9; idx += 1

    VALID_COORDS[idx][0] = 7; VALID_COORDS[idx][1] = 3; idx += 1
    VALID_COORDS[idx][0] = 7; VALID_COORDS[idx][1] = 4; idx += 1
    VALID_COORDS[idx][0] = 7; VALID_COORDS[idx][1] = 5; idx += 1
    VALID_COORDS[idx][0] = 7; VALID_COORDS[idx][1] = 6; idx += 1
    VALID_COORDS[idx][0] = 7; VALID_COORDS[idx][1] = 7; idx += 1
    VALID_COORDS[idx][0] = 7; VALID_COORDS[idx][1] = 8; idx += 1
    VALID_COORDS[idx][0] = 7; VALID_COORDS[idx][1] = 9; idx += 1

    VALID_COORDS[idx][0] = 6; VALID_COORDS[idx][1] = 2; idx += 1
    VALID_COORDS[idx][0] = 6; VALID_COORDS[idx][1] = 3; idx += 1
    VALID_COORDS[idx][0] = 6; VALID_COORDS[idx][1] = 4; idx += 1
    VALID_COORDS[idx][0] = 6; VALID_COORDS[idx][1] = 5; idx += 1
    VALID_COORDS[idx][0] = 6; VALID_COORDS[idx][1] = 6; idx += 1
    VALID_COORDS[idx][0] = 6; VALID_COORDS[idx][1] = 7; idx += 1
    VALID_COORDS[idx][0] = 6; VALID_COORDS[idx][1] = 8; idx += 1
    VALID_COORDS[idx][0] = 6; VALID_COORDS[idx][1] = 9; idx += 1

    VALID_COORDS[idx][0] = 5; VALID_COORDS[idx][1] = 1; idx += 1
    VALID_COORDS[idx][0] = 5; VALID_COORDS[idx][1] = 2; idx += 1
    VALID_COORDS[idx][0] = 5; VALID_COORDS[idx][1] = 3; idx += 1
    VALID_COORDS[idx][0] = 5; VALID_COORDS[idx][1] = 4; idx += 1
    VALID_COORDS[idx][0] = 5; VALID_COORDS[idx][1] = 5; idx += 1
    VALID_COORDS[idx][0] = 5; VALID_COORDS[idx][1] = 6; idx += 1
    VALID_COORDS[idx][0] = 5; VALID_COORDS[idx][1] = 7; idx += 1
    VALID_COORDS[idx][0] = 5; VALID_COORDS[idx][1] = 8; idx += 1
    VALID_COORDS[idx][0] = 5; VALID_COORDS[idx][1] = 9; idx += 1

    VALID_COORDS[idx][0] = 4; VALID_COORDS[idx][1] = 1; idx += 1
    VALID_COORDS[idx][0] = 4; VALID_COORDS[idx][1] = 2; idx += 1
    VALID_COORDS[idx][0] = 4; VALID_COORDS[idx][1] = 3; idx += 1
    VALID_COORDS[idx][0] = 4; VALID_COORDS[idx][1] = 4; idx += 1
    VALID_COORDS[idx][0] = 4; VALID_COORDS[idx][1] = 5; idx += 1
    VALID_COORDS[idx][0] = 4; VALID_COORDS[idx][1] = 6; idx += 1
    VALID_COORDS[idx][0] = 4; VALID_COORDS[idx][1] = 7; idx += 1
    VALID_COORDS[idx][0] = 4; VALID_COORDS[idx][1] = 8; idx += 1

    VALID_COORDS[idx][0] = 3; VALID_COORDS[idx][1] = 1; idx += 1
    VALID_COORDS[idx][0] = 3; VALID_COORDS[idx][1] = 2; idx += 1
    VALID_COORDS[idx][0] = 3; VALID_COORDS[idx][1] = 3; idx += 1
    VALID_COORDS[idx][0] = 3; VALID_COORDS[idx][1] = 4; idx += 1
    VALID_COORDS[idx][0] = 3; VALID_COORDS[idx][1] = 5; idx += 1
    VALID_COORDS[idx][0] = 3; VALID_COORDS[idx][1] = 6; idx += 1
    VALID_COORDS[idx][0] = 3; VALID_COORDS[idx][1] = 7; idx += 1

    VALID_COORDS[idx][0] = 2; VALID_COORDS[idx][1] = 1; idx += 1
    VALID_COORDS[idx][0] = 2; VALID_COORDS[idx][1] = 2; idx += 1
    VALID_COORDS[idx][0] = 2; VALID_COORDS[idx][1] = 3; idx += 1
    VALID_COORDS[idx][0] = 2; VALID_COORDS[idx][1] = 4; idx += 1
    VALID_COORDS[idx][0] = 2; VALID_COORDS[idx][1] = 5; idx += 1
    VALID_COORDS[idx][0] = 2; VALID_COORDS[idx][1] = 6; idx += 1

    VALID_COORDS[idx][0] = 1; VALID_COORDS[idx][1] = 1; idx += 1
    VALID_COORDS[idx][0] = 1; VALID_COORDS[idx][1] = 2; idx += 1
    VALID_COORDS[idx][0] = 1; VALID_COORDS[idx][1] = 3; idx += 1
    VALID_COORDS[idx][0] = 1; VALID_COORDS[idx][1] = 4; idx += 1
    VALID_COORDS[idx][0] = 1; VALID_COORDS[idx][1] = 5; idx += 1

VALID_COORDS_NP = np.array([
    [9, 5], [9, 6], [9, 7], [9, 8], [9, 9],
    [8, 4], [8, 5], [8, 6], [8, 7], [8, 8], [8, 9],
    [7, 3], [7, 4], [7, 5], [7, 6], [7, 7], [7, 8], [7, 9],
    [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8], [6, 9],
    [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8], [5, 9],
    [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8],
    [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7],
    [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6],
    [1, 1], [1, 2], [1, 3], [1, 4], [1, 5]
], dtype=np.int64)

VALID_COORDS_SET = {tuple(coord) for coord in VALID_COORDS_NP}

cdef int VALID_COORDS_LOOKUP[10][10]

cdef void init_valid_coords_lookup():
    cdef int i, j, row, col

    for i in range(10):
        for j in range(10):
            VALID_COORDS_LOOKUP[i][j] = 0

    for i in range(VALID_COORDS_COUNT):
        row = VALID_COORDS[i][0]
        col = VALID_COORDS[i][1]
        VALID_COORDS_LOOKUP[row][col] = 1

cdef double CENTRALITY_MAP_C[10][10]
cdef double CENTER_SCORE = 6.7
cdef double RING1_SCORE = 4.5
cdef double RING2_SCORE = 2.8
cdef double RING3_SCORE = 1.3
cdef double RING4_SCORE = -1.3

cdef void init_centrality_map_c():
    cdef int i, j

    for i in range(10):
        for j in range(10):
            CENTRALITY_MAP_C[i][j] = 1.0

    CENTRALITY_MAP_C[5][5] = CENTER_SCORE

    CENTRALITY_MAP_C[6][5] = RING1_SCORE
    CENTRALITY_MAP_C[6][6] = RING1_SCORE
    CENTRALITY_MAP_C[5][4] = RING1_SCORE
    CENTRALITY_MAP_C[5][6] = RING1_SCORE
    CENTRALITY_MAP_C[4][4] = RING1_SCORE
    CENTRALITY_MAP_C[4][5] = RING1_SCORE

    CENTRALITY_MAP_C[7][5] = RING2_SCORE
    CENTRALITY_MAP_C[7][6] = RING2_SCORE
    CENTRALITY_MAP_C[7][7] = RING2_SCORE
    CENTRALITY_MAP_C[6][4] = RING2_SCORE
    CENTRALITY_MAP_C[6][7] = RING2_SCORE
    CENTRALITY_MAP_C[5][3] = RING2_SCORE
    CENTRALITY_MAP_C[5][7] = RING2_SCORE
    CENTRALITY_MAP_C[4][3] = RING2_SCORE
    CENTRALITY_MAP_C[4][6] = RING2_SCORE
    CENTRALITY_MAP_C[3][3] = RING2_SCORE
    CENTRALITY_MAP_C[3][4] = RING2_SCORE
    CENTRALITY_MAP_C[3][5] = RING2_SCORE

    CENTRALITY_MAP_C[8][5] = RING3_SCORE
    CENTRALITY_MAP_C[8][6] = RING3_SCORE
    CENTRALITY_MAP_C[8][7] = RING3_SCORE
    CENTRALITY_MAP_C[8][8] = RING3_SCORE
    CENTRALITY_MAP_C[7][4] = RING3_SCORE
    CENTRALITY_MAP_C[7][8] = RING3_SCORE
    CENTRALITY_MAP_C[6][3] = RING3_SCORE
    CENTRALITY_MAP_C[6][8] = RING3_SCORE
    CENTRALITY_MAP_C[5][2] = RING3_SCORE
    CENTRALITY_MAP_C[5][8] = RING3_SCORE
    CENTRALITY_MAP_C[4][2] = RING3_SCORE
    CENTRALITY_MAP_C[4][7] = RING3_SCORE
    CENTRALITY_MAP_C[3][2] = RING3_SCORE
    CENTRALITY_MAP_C[3][6] = RING3_SCORE
    CENTRALITY_MAP_C[2][2] = RING3_SCORE
    CENTRALITY_MAP_C[2][3] = RING3_SCORE
    CENTRALITY_MAP_C[2][4] = RING3_SCORE
    CENTRALITY_MAP_C[2][5] = RING3_SCORE

    CENTRALITY_MAP_C[9][5] = RING4_SCORE
    CENTRALITY_MAP_C[9][6] = RING4_SCORE
    CENTRALITY_MAP_C[9][7] = RING4_SCORE
    CENTRALITY_MAP_C[9][8] = RING4_SCORE
    CENTRALITY_MAP_C[9][9] = RING4_SCORE
    CENTRALITY_MAP_C[8][4] = RING4_SCORE
    CENTRALITY_MAP_C[8][9] = RING4_SCORE
    CENTRALITY_MAP_C[7][3] = RING4_SCORE
    CENTRALITY_MAP_C[7][9] = RING4_SCORE
    CENTRALITY_MAP_C[6][2] = RING4_SCORE
    CENTRALITY_MAP_C[6][9] = RING4_SCORE
    CENTRALITY_MAP_C[5][1] = RING4_SCORE
    CENTRALITY_MAP_C[5][9] = RING4_SCORE
    CENTRALITY_MAP_C[4][1] = RING4_SCORE
    CENTRALITY_MAP_C[4][8] = RING4_SCORE
    CENTRALITY_MAP_C[3][1] = RING4_SCORE
    CENTRALITY_MAP_C[3][7] = RING4_SCORE
    CENTRALITY_MAP_C[2][1] = RING4_SCORE
    CENTRALITY_MAP_C[2][6] = RING4_SCORE
    CENTRALITY_MAP_C[1][1] = RING4_SCORE
    CENTRALITY_MAP_C[1][2] = RING4_SCORE
    CENTRALITY_MAP_C[1][3] = RING4_SCORE
    CENTRALITY_MAP_C[1][4] = RING4_SCORE
    CENTRALITY_MAP_C[1][5] = RING4_SCORE

CENTER_COORDS_NP = np.array([(5, 5)], dtype=np.int64)

RING1_COORDS_NP = np.array([
    (6, 5), (6, 6), (5, 4), (5, 6), (4, 4), (4, 5)
], dtype=np.int64)

RING2_COORDS_NP = np.array([
    (7, 5), (7, 6), (7, 7), (6, 4), (6, 7), (5, 3),
    (5, 7), (4, 3), (4, 6), (3, 3), (3, 4), (3, 5)
], dtype=np.int64)

RING3_COORDS_NP = np.array([
    (8, 5), (8, 6), (8, 7), (8, 8), (7, 4), (7, 8),
    (6, 3), (6, 8), (5, 2), (5, 8), (4, 2), (4, 7),
    (3, 2), (3, 6), (2, 2), (2, 3), (2, 4), (2, 5)
], dtype=np.int64)

RING4_COORDS_NP = np.array([
    (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (8, 4),
    (8, 9), (7, 3), (7, 9), (6, 2), (6, 9), (5, 1),
    (5, 9), (4, 1), (4, 8), (3, 1), (3, 7), (2, 1),
    (2, 6), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)
], dtype=np.int64)

CENTRALITY_MAP = {}
for coord in CENTER_COORDS_NP:
    CENTRALITY_MAP[tuple(coord)] = CENTER_SCORE
for coord in RING1_COORDS_NP:
    CENTRALITY_MAP[tuple(coord)] = RING1_SCORE
for coord in RING2_COORDS_NP:
    CENTRALITY_MAP[tuple(coord)] = RING2_SCORE
for coord in RING3_COORDS_NP:
    CENTRALITY_MAP[tuple(coord)] = RING3_SCORE
for coord in RING4_COORDS_NP:
    CENTRALITY_MAP[tuple(coord)] = RING4_SCORE

cdef int OUTER_RING_COORDS[24][2]
cdef int MIDDLE_RING_COORDS[18][2]
cdef int INNER_RING_COORDS[12][2]

cdef void init_ring_coords():

    OUTER_RING_COORDS[0][0] = 9; OUTER_RING_COORDS[0][1] = 5
    OUTER_RING_COORDS[1][0] = 9; OUTER_RING_COORDS[1][1] = 6
    OUTER_RING_COORDS[2][0] = 9; OUTER_RING_COORDS[2][1] = 7
    OUTER_RING_COORDS[3][0] = 9; OUTER_RING_COORDS[3][1] = 8
    OUTER_RING_COORDS[4][0] = 9; OUTER_RING_COORDS[4][1] = 9
    OUTER_RING_COORDS[5][0] = 8; OUTER_RING_COORDS[5][1] = 4
    OUTER_RING_COORDS[6][0] = 8; OUTER_RING_COORDS[6][1] = 9
    OUTER_RING_COORDS[7][0] = 7; OUTER_RING_COORDS[7][1] = 3
    OUTER_RING_COORDS[8][0] = 7; OUTER_RING_COORDS[8][1] = 9
    OUTER_RING_COORDS[9][0] = 6; OUTER_RING_COORDS[9][1] = 2
    OUTER_RING_COORDS[10][0] = 6; OUTER_RING_COORDS[10][1] = 9
    OUTER_RING_COORDS[11][0] = 5; OUTER_RING_COORDS[11][1] = 1
    OUTER_RING_COORDS[12][0] = 5; OUTER_RING_COORDS[12][1] = 9
    OUTER_RING_COORDS[13][0] = 4; OUTER_RING_COORDS[13][1] = 1
    OUTER_RING_COORDS[14][0] = 4; OUTER_RING_COORDS[14][1] = 8
    OUTER_RING_COORDS[15][0] = 3; OUTER_RING_COORDS[15][1] = 1
    OUTER_RING_COORDS[16][0] = 3; OUTER_RING_COORDS[16][1] = 7
    OUTER_RING_COORDS[17][0] = 2; OUTER_RING_COORDS[17][1] = 1
    OUTER_RING_COORDS[18][0] = 2; OUTER_RING_COORDS[18][1] = 6
    OUTER_RING_COORDS[19][0] = 1; OUTER_RING_COORDS[19][1] = 1
    OUTER_RING_COORDS[20][0] = 1; OUTER_RING_COORDS[20][1] = 2
    OUTER_RING_COORDS[21][0] = 1; OUTER_RING_COORDS[21][1] = 3
    OUTER_RING_COORDS[22][0] = 1; OUTER_RING_COORDS[22][1] = 4
    OUTER_RING_COORDS[23][0] = 1; OUTER_RING_COORDS[23][1] = 5

    MIDDLE_RING_COORDS[0][0] = 8; MIDDLE_RING_COORDS[0][1] = 5
    MIDDLE_RING_COORDS[1][0] = 8; MIDDLE_RING_COORDS[1][1] = 6
    MIDDLE_RING_COORDS[2][0] = 8; MIDDLE_RING_COORDS[2][1] = 7
    MIDDLE_RING_COORDS[3][0] = 8; MIDDLE_RING_COORDS[3][1] = 8
    MIDDLE_RING_COORDS[4][0] = 7; MIDDLE_RING_COORDS[4][1] = 4
    MIDDLE_RING_COORDS[5][0] = 7; MIDDLE_RING_COORDS[5][1] = 8
    MIDDLE_RING_COORDS[6][0] = 6; MIDDLE_RING_COORDS[6][1] = 3
    MIDDLE_RING_COORDS[7][0] = 6; MIDDLE_RING_COORDS[7][1] = 8
    MIDDLE_RING_COORDS[8][0] = 5; MIDDLE_RING_COORDS[8][1] = 2
    MIDDLE_RING_COORDS[9][0] = 5; MIDDLE_RING_COORDS[9][1] = 8
    MIDDLE_RING_COORDS[10][0] = 4; MIDDLE_RING_COORDS[10][1] = 2
    MIDDLE_RING_COORDS[11][0] = 4; MIDDLE_RING_COORDS[11][1] = 7
    MIDDLE_RING_COORDS[12][0] = 3; MIDDLE_RING_COORDS[12][1] = 2
    MIDDLE_RING_COORDS[13][0] = 3; MIDDLE_RING_COORDS[13][1] = 6
    MIDDLE_RING_COORDS[14][0] = 2; MIDDLE_RING_COORDS[14][1] = 2
    MIDDLE_RING_COORDS[15][0] = 2; MIDDLE_RING_COORDS[15][1] = 3
    MIDDLE_RING_COORDS[16][0] = 2; MIDDLE_RING_COORDS[16][1] = 4
    MIDDLE_RING_COORDS[17][0] = 2; MIDDLE_RING_COORDS[17][1] = 5

    INNER_RING_COORDS[0][0] = 7; INNER_RING_COORDS[0][1] = 5
    INNER_RING_COORDS[1][0] = 7; INNER_RING_COORDS[1][1] = 6
    INNER_RING_COORDS[2][0] = 7; INNER_RING_COORDS[2][1] = 7
    INNER_RING_COORDS[3][0] = 6; INNER_RING_COORDS[3][1] = 4
    INNER_RING_COORDS[4][0] = 6; INNER_RING_COORDS[4][1] = 7
    INNER_RING_COORDS[5][0] = 5; INNER_RING_COORDS[5][1] = 3
    INNER_RING_COORDS[6][0] = 5; INNER_RING_COORDS[6][1] = 7
    INNER_RING_COORDS[7][0] = 4; INNER_RING_COORDS[7][1] = 3
    INNER_RING_COORDS[8][0] = 4; INNER_RING_COORDS[8][1] = 6
    INNER_RING_COORDS[9][0] = 3; INNER_RING_COORDS[9][1] = 3
    INNER_RING_COORDS[10][0] = 3; INNER_RING_COORDS[10][1] = 4
    INNER_RING_COORDS[11][0] = 3; INNER_RING_COORDS[11][1] = 5

NEIGHBOR_CACHE = {}
for i in range(VALID_COORDS_NP.shape[0]):
    coord = tuple(VALID_COORDS_NP[i])
    neighbors = []
    for j in range(DIRECTIONS_NP.shape[0]):
        neighbor = (VALID_COORDS_NP[i, 0] + DIRECTIONS_NP[j, 0], VALID_COORDS_NP[i, 1] + DIRECTIONS_NP[j, 1])
        if neighbor in VALID_COORDS_SET:
            neighbors.append(neighbor)
    NEIGHBOR_CACHE[coord] = neighbors

COORD_TO_INDEX_MAP = {}
for i in range(VALID_COORDS_NP.shape[0]):
    COORD_TO_INDEX_MAP[tuple(VALID_COORDS_NP[i])] = i

cdef int COORD_TO_INDEX_MAP_C[10][10]

cdef void init_coord_to_index_map_c():
    cdef int i, j, row, col

    for i in range(10):
        for j in range(10):
            COORD_TO_INDEX_MAP_C[i][j] = -1

    for i in range(VALID_COORDS_COUNT):
        row = VALID_COORDS[i][0]
        col = VALID_COORDS[i][1]
        COORD_TO_INDEX_MAP_C[row][col] = i

cdef uint64_t ZOBRIST_TABLE_C[2][61]

cdef void init_zobrist_table_c():
    cdef int i, j, seed = 42  # Use fixed seed for deterministic results

    srand(seed)

    for i in range(2):  # 0 for black, 1 for white
        for j in range(61):  # For each valid board position
            ZOBRIST_TABLE_C[i][j] = rand() << 32 | rand()


np.random.seed(42)
ZOBRIST_TABLE_NP = np.random.randint(
    0, 2 ** 64 - 1, size=(2, VALID_COORDS_NP.shape[0]), dtype=np.uint64
)

np.random.seed(None)

cdef void init_tt_table():
    global tt_table

    if tt_table != NULL:
        free(tt_table)

    tt_table = <TTEntry *> calloc(TT_SIZE, sizeof(TTEntry))

    if tt_table == NULL:
        raise MemoryError("Transposition memory assignment failed")

cdef void tt_store(uint64_t hash_key, int depth, double score, int flag, uint64_t best_move_hash, int age):
    cdef int index = hash_key & TT_MASK
    cdef TTEntry * entry = &tt_table[index]

    if (entry.hash_key == 0 or
            depth > entry.depth or
            age > entry.age):
        entry.hash_key = hash_key
        entry.depth = depth
        entry.score = score
        entry.flag = flag
        entry.best_move_hash = best_move_hash
        entry.age = age

cdef bint tt_probe(uint64_t hash_key, int depth, double * score, int * flag, uint64_t * best_move_hash):
    cdef int index = hash_key & TT_MASK
    cdef TTEntry * entry = &tt_table[index]

    if entry.hash_key == hash_key:
        score[0] = entry.score
        flag[0] = entry.flag
        best_move_hash[0] = entry.best_move_hash

        return entry.depth >= depth  # Return True if entry has sufficient depth

    return False

cdef int tt_age = 0

cdef void clear_tt():
    global tt_age

    if tt_table != NULL:
        memset(tt_table, 0, TT_SIZE * sizeof(TTEntry))

    tt_age += 1

cdef void initialize_all():
    init_directions()
    init_valid_coords()
    init_valid_coords_lookup()
    init_centrality_map_c()
    init_ring_coords()
    init_zobrist_table_c()
    init_coord_to_index_map_c()
    init_tt_table()
    init_default_weights()

initialize_all()

history_table = {}
killer_moves = {}
group_cache = OrderedDict()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void manage_cache_size(object cache, int max_size):
    
    if len(cache) > max_size:
        remove_count = max_size // 5
        for _ in range(remove_count):
            if cache:
                cache.popitem(last=False)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint is_valid_coord_c(int row, int col):
    
    return 0 <= row < 10 and 0 <= col < 10 and VALID_COORDS_LOOKUP[row][col] == 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint is_valid_coord(tuple coord):
    
    return coord in VALID_COORDS_SET

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline list get_neighbors(tuple coord):
    
    return NEIGHBOR_CACHE.get(coord, [])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int coord_to_index(tuple coord):
    
    return COORD_TO_INDEX_MAP.get(coord, -1)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int coord_to_index_c(int row, int col):
    
    if 0 <= row < 10 and 0 <= col < 10:
        return COORD_TO_INDEX_MAP_C[row][col]
    return -1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void copy_marbles_to_c_array(list marbles_list, int * rows, int * cols, int * count):
    
    count[0] = len(marbles_list)
    cdef int i
    for i in range(count[0]):
        rows[i] = marbles_list[i][0]
        cols[i] = marbles_list[i][1]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void create_board_lookup(int * friend_rows, int * friend_cols, int friend_count,
                              int * enemy_rows, int * enemy_cols, int enemy_count,
                              int * board_lookup):
    
    cdef int i, idx

    for i in range(MAX_BOARD_SIZE * MAX_BOARD_SIZE):
        board_lookup[i] = 0

    for i in range(friend_count):
        idx = friend_rows[i] * MAX_BOARD_SIZE + friend_cols[i]
        if 0 <= idx < MAX_BOARD_SIZE * MAX_BOARD_SIZE:
            board_lookup[idx] = 1

    for i in range(enemy_count):
        idx = enemy_rows[i] * MAX_BOARD_SIZE + enemy_cols[i]
        if 0 <= idx < MAX_BOARD_SIZE * MAX_BOARD_SIZE:
            board_lookup[idx] = 2

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint is_player_marble(int row, int col, int * board_lookup):
    
    cdef int idx = row * MAX_BOARD_SIZE + col
    if 0 <= idx < MAX_BOARD_SIZE * MAX_BOARD_SIZE:
        return board_lookup[idx] == 1
    return False

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint is_enemy_marble(int row, int col, int * board_lookup):
    
    cdef int idx = row * MAX_BOARD_SIZE + col
    if 0 <= idx < MAX_BOARD_SIZE * MAX_BOARD_SIZE:
        return board_lookup[idx] == 2
    return False

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint is_in_outer_ring(int row, int col):
    
    cdef int i
    for i in range(24):
        if OUTER_RING_COORDS[i][0] == row and OUTER_RING_COORDS[i][1] == col:
            return True
    return False

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint is_in_middle_ring(int row, int col):
    
    cdef int i
    for i in range(18):
        if MIDDLE_RING_COORDS[i][0] == row and MIDDLE_RING_COORDS[i][1] == col:
            return True
    return False

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint is_in_inner_ring(int row, int col):
    
    cdef int i
    for i in range(12):
        if INNER_RING_COORDS[i][0] == row and INNER_RING_COORDS[i][1] == col:
            return True
    return False

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void find_groups_c(int * rows, int * cols, int count, int * board_lookup, CGroup * groups, int * group_count):
    
    cdef int i, j, k, idx
    cdef bint is_adjacent
    cdef int dr, dc, dr2, dc2
    cdef int next_row, next_col

    group_count[0] = 0

    for i in range(count):
        if group_count[0] < MAX_GROUPS:
            groups[group_count[0]].marbles[0].row = rows[i]
            groups[group_count[0]].marbles[0].col = cols[i]
            groups[group_count[0]].size = 1
            group_count[0] += 1

    for i in range(count):
        for j in range(i + 1, count):
            dr = rows[j] - rows[i]
            dc = cols[j] - cols[i]

            is_adjacent = False
            for k in range(DIRECTIONS_COUNT):
                if (dr == DIRECTIONS[k][0] and dc == DIRECTIONS[k][1]) or \
                        (dr == -DIRECTIONS[k][0] and dc == -DIRECTIONS[k][1]):
                    is_adjacent = True
                    break

            if is_adjacent and group_count[0] < MAX_GROUPS:
                groups[group_count[0]].marbles[0].row = rows[i]
                groups[group_count[0]].marbles[0].col = cols[i]
                groups[group_count[0]].marbles[1].row = rows[j]
                groups[group_count[0]].marbles[1].col = cols[j]
                groups[group_count[0]].size = 2
                group_count[0] += 1

    cdef int max_twos = min(group_count[0], 50)  # Limit search for efficiency
    for i in range(max_twos):
        if groups[i].size == 2:
            dr = groups[i].marbles[1].row - groups[i].marbles[0].row
            dc = groups[i].marbles[1].col - groups[i].marbles[0].col

            next_row = groups[i].marbles[1].row + dr
            next_col = groups[i].marbles[1].col + dc

            if is_player_marble(next_row, next_col, board_lookup) and group_count[0] < MAX_GROUPS:
                groups[group_count[0]].marbles[0].row = groups[i].marbles[0].row
                groups[group_count[0]].marbles[0].col = groups[i].marbles[0].col
                groups[group_count[0]].marbles[1].row = groups[i].marbles[1].row
                groups[group_count[0]].marbles[1].col = groups[i].marbles[1].col
                groups[group_count[0]].marbles[2].row = next_row
                groups[group_count[0]].marbles[2].col = next_col
                groups[group_count[0]].size = 3
                group_count[0] += 1

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double evaluate_marble_difference(int friend_count, int enemy_count) nogil:
    cdef double score

    score = 100 * (friend_count - enemy_count)

    if friend_count <= 8:
        score = -10000.0
    elif enemy_count <= 8:
        score = 10000.0

    return score

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double evaluate_centrality(int *friend_rows, int *friend_cols, int friend_count,
                                       int *enemy_rows, int *enemy_cols, int enemy_count) nogil:
    
    cdef double friend_centrality = 0.0
    cdef double enemy_centrality = 0.0
    cdef double friend_weight = 1.0
    cdef double enemy_weight = 1.0
    cdef double scale_factor = 1.0
    cdef double score
    cdef int i, row, col

    for i in range(friend_count):
        row = friend_rows[i]
        col = friend_cols[i]
        if 0 <= row < 10 and 0 <= col < 10:
            score = CENTRALITY_MAP_C[row][col] * scale_factor
            friend_centrality += score

    for i in range(enemy_count):
        row = enemy_rows[i]
        col = enemy_cols[i]
        if 0 <= row < 10 and 0 <= col < 10:
            score = CENTRALITY_MAP_C[row][col] * scale_factor
            enemy_centrality += score

    if friend_count < 14:
        friend_weight = 1.0 - (14 - friend_count) * 0.03

    if enemy_count < 14:
        enemy_weight = 1.0 - (14 - enemy_count) * 0.03

    if friend_count > 0:
        friend_centrality = friend_centrality * (friend_count * friend_weight)

    if enemy_count > 0:
        enemy_centrality = enemy_centrality * (enemy_count * enemy_weight)

    return friend_centrality - enemy_centrality

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint is_in_middle_ring_direct(int row, int col) nogil:
    
    return ((row == 8 and col == 5) or (row == 8 and col == 6) or
            (row == 8 and col == 7) or (row == 8 and col == 8) or
            (row == 7 and col == 4) or (row == 7 and col == 8) or
            (row == 6 and col == 3) or (row == 6 and col == 8) or
            (row == 5 and col == 2) or (row == 5 and col == 8) or
            (row == 4 and col == 2) or (row == 4 and col == 7) or
            (row == 3 and col == 2) or (row == 3 and col == 6) or
            (row == 2 and col == 2) or (row == 2 and col == 3) or
            (row == 2 and col == 4) or (row == 2 and col == 5))

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint is_in_outer_ring_direct(int row, int col) nogil:
    
    return ((row == 9 and col == 5) or (row == 9 and col == 6) or
            (row == 9 and col == 7) or (row == 9 and col == 8) or
            (row == 9 and col == 9) or (row == 8 and col == 4) or
            (row == 8 and col == 9) or (row == 7 and col == 3) or
            (row == 7 and col == 9) or (row == 6 and col == 2) or
            (row == 6 and col == 9) or (row == 5 and col == 1) or
            (row == 5 and col == 9) or (row == 4 and col == 1) or
            (row == 4 and col == 8) or (row == 3 and col == 1) or
            (row == 3 and col == 7) or (row == 2 and col == 1) or
            (row == 2 and col == 6) or (row == 1 and col == 1) or
            (row == 1 and col == 2) or (row == 1 and col == 3) or
            (row == 1 and col == 4) or (row == 1 and col == 5))

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double evaluate_push_ability(int *board_lookup) nogil:
    cdef double push_score = 0.0
    cdef int row, col, idx

    for row in range(1, 10):
        for col in range(1, 10):
            idx = row * MAX_BOARD_SIZE + col
            if 0 <= idx < MAX_BOARD_SIZE * MAX_BOARD_SIZE and board_lookup[idx] == 2:

                if is_in_middle_ring_direct(row, col):
                    push_score += 1.2

                elif is_in_outer_ring_direct(row, col):
                    push_score += 2.5

    return push_score

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double evaluate_formation(int *friend_rows, int *friend_cols, int friend_count,
                                      int *board_lookup) nogil:
    
    cdef double formation_score = 0.0
    cdef int i, dir_idx, connection_count
    cdef int neighbor_row, neighbor_col, idx

    if friend_count >= 3:
        for i in range(friend_count):
            connection_count = 0

            for dir_idx in range(DIRECTIONS_COUNT):
                neighbor_row = friend_rows[i] + DIRECTIONS[dir_idx][0]
                neighbor_col = friend_cols[i] + DIRECTIONS[dir_idx][1]

                idx = neighbor_row * MAX_BOARD_SIZE + neighbor_col
                if 0 <= neighbor_row < 10 and 0 <= neighbor_col < 10 and 0 <= idx < MAX_BOARD_SIZE * MAX_BOARD_SIZE and \
                        board_lookup[idx] == 1:
                    connection_count += 1

            if connection_count == 2:
                formation_score += 1.0
            elif connection_count == 3:
                formation_score += 1.25
            elif connection_count == 4:
                formation_score += 1.55
            elif connection_count == 5:
                formation_score += 1.27
            elif connection_count == 6:
                formation_score += 1.3

    return formation_score * 0.8

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double evaluate_connectivity(int *friend_rows, int *friend_cols, int friend_count,
                                         int *board_lookup) nogil:
    
    cdef double connectivity_score = 0.0
    cdef int i, dir_idx, connection_count
    cdef int neighbor_row, neighbor_col, idx

    if friend_count >= 3:
        for i in range(friend_count):
            connection_count = 0

            for dir_idx in range(DIRECTIONS_COUNT):
                neighbor_row = friend_rows[i] + DIRECTIONS[dir_idx][0]
                neighbor_col = friend_cols[i] + DIRECTIONS[dir_idx][1]

                idx = neighbor_row * MAX_BOARD_SIZE + neighbor_col
                if 0 <= neighbor_row < 10 and 0 <= neighbor_col < 10 and 0 <= idx < MAX_BOARD_SIZE * MAX_BOARD_SIZE and \
                        board_lookup[idx] == 1:
                    connection_count += 1

            if connection_count == 0:
                connectivity_score -= 7.5

            elif connection_count == 1:
                connectivity_score -= 4.0

    return connectivity_score * 0.9

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double evaluate_board_with_features_c(int *friend_rows, int *friend_cols, int friend_count,
                                           int *enemy_rows, int *enemy_cols, int enemy_count,
                                           int *board_lookup, CGroup *groups, int group_count,
                                           EvaluationWeights *weights) nogil:
    
    cdef double scores[5]
    cdef int feature_idx
    cdef int num_features = 5

    for feature_idx in prange(num_features, nogil=True, schedule='static'):
        if feature_idx == 0:

            scores[0] = evaluate_marble_difference(friend_count, enemy_count)
        elif feature_idx == 1:

            scores[1] = evaluate_centrality(friend_rows, friend_cols, friend_count,
                                            enemy_rows, enemy_cols, enemy_count)
        elif feature_idx == 2:

            scores[2] = evaluate_push_ability(board_lookup)
        elif feature_idx == 3:

            scores[3] = evaluate_formation(friend_rows, friend_cols, friend_count, board_lookup)
        elif feature_idx == 4:

            scores[4] = evaluate_connectivity(friend_rows, friend_cols, friend_count, board_lookup)

    scores[0] *= weights.marble_diff
    scores[1] *= weights.centrality
    scores[2] *= weights.push_ability
    scores[3] *= weights.formation
    scores[4] *= weights.connectivity

    return scores[0] + scores[1] + scores[2] + scores[3] + scores[4]

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def evaluate_board_with_features(list board, str player):
    

    cdef int friend_idx = 0 if player.lower() == "black" else 1
    cdef int enemy_idx = 1 if player.lower() == "black" else 0

    cdef list friend_marbles = board[friend_idx]
    cdef list enemy_marbles = board[enemy_idx]

    cdef int friend_count = len(friend_marbles)
    cdef int enemy_count = len(enemy_marbles)

    cdef int * friend_rows = <int *> malloc(MAX_MARBLES * sizeof(int))
    cdef int * friend_cols = <int *> malloc(MAX_MARBLES * sizeof(int))
    cdef int * enemy_rows = <int *> malloc(MAX_MARBLES * sizeof(int))
    cdef int * enemy_cols = <int *> malloc(MAX_MARBLES * sizeof(int))
    cdef int * board_lookup = <int *> malloc(MAX_BOARD_SIZE * MAX_BOARD_SIZE * sizeof(int))
    cdef CGroup * groups = <CGroup *> malloc(MAX_GROUPS * sizeof(CGroup))
    cdef int group_count = 0
    cdef double total_score

    if not friend_rows or not friend_cols or not enemy_rows or not enemy_cols or not board_lookup or not groups:
        if friend_rows: free(friend_rows)
        if friend_cols: free(friend_cols)
        if enemy_rows: free(enemy_rows)
        if enemy_cols: free(enemy_cols)
        if board_lookup: free(board_lookup)
        if groups: free(groups)
        raise MemoryError("memory issue in evaluate_board_with_features")
    try:

        copy_marbles_to_c_array(friend_marbles, friend_rows, friend_cols, &friend_count)
        copy_marbles_to_c_array(enemy_marbles, enemy_rows, enemy_cols, &enemy_count)

        create_board_lookup(friend_rows, friend_cols, friend_count, enemy_rows, enemy_cols, enemy_count, board_lookup)

        find_groups_c(friend_rows, friend_cols, friend_count, board_lookup, groups, &group_count)

        total_score = evaluate_board_with_features_c(
            friend_rows, friend_cols, friend_count,
            enemy_rows, enemy_cols, enemy_count,
            board_lookup, groups, group_count,
            &DEFAULT_WEIGHTS
        )

        return total_score

    finally:

        if friend_rows: free(friend_rows)
        if friend_cols: free(friend_cols)
        if enemy_rows: free(enemy_rows)
        if enemy_cols: free(enemy_cols)
        if board_lookup: free(board_lookup)
        if groups: free(groups)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def evaluate_board(list board, str player):
    
    return evaluate_board_with_features(board, player)


@cython.boundscheck(False)
@cython.wraparound(False)
def move_ordering_score(list move, int depth, tuple prev_best=None):
    
    cdef double score = 0.0
    cdef uint64_t move_key = compute_zobrist_hash(move)
    cdef str move_key_str = str(move_key)

    if prev_best and move == prev_best[1]:
        return float('inf')

    if depth in killer_moves and move_key_str in killer_moves[depth]:
        score += 100.0 + float(killer_moves[depth][move_key_str])

    return score

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _enhanced_move_ordering(tuple item, int depth, tuple prev_best):
    
    cdef object move_key, move
    move_key, move = item
    return move_ordering_score(move, depth, prev_best)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple alpha_beta_with_time_check(list board, int depth, double alpha, double beta, str player,
                                      str maximizing_player,
                                      object move_generator, double time_start, double time_limit,
                                      tuple prev_best=None):
    
    cdef double current_time
    cdef uint64_t board_hash
    cdef double eval_score, best_score, score
    cdef list moves_items
    cdef str next_player, current_color
    cdef object best_move = None
    cdef object best_move_key = None
    cdef bint is_maximizer
    cdef uint64_t move_hash
    cdef str move_hash_str
    cdef int tt_flag
    cdef uint64_t best_move_hash = 0
    cdef bint tt_hit

    if depth % 2 == 0:
        current_time = time.time()
        if current_time - time_start > time_limit:
            raise TimeoutError("Search time limit exceeded")

    board_hash = compute_zobrist_hash(board)

    cdef double tt_score = 0
    cdef int tt_entry_flag = 0
    cdef uint64_t tt_best_move_hash = 0

    tt_hit = tt_probe(board_hash, depth, &tt_score, &tt_entry_flag, &tt_best_move_hash)

    if tt_hit:
        if tt_entry_flag == TT_EXACT:

            return (tt_score, None, None)
        elif tt_entry_flag == TT_LOWER and tt_score >= beta:

            return (tt_score, None, None)
        elif tt_entry_flag == TT_UPPER and tt_score <= alpha:

            return (tt_score, None, None)

    if depth <= 0:
        eval_score = evaluate_board(board, maximizing_player)

        tt_store(board_hash, depth, eval_score, TT_EXACT, 0, tt_age)
        return (eval_score, None, None)

    current_color = "BLACK" if player.lower() == "black" else "WHITE"
    moves_dict = move_generator(board, current_color)

    if not moves_dict:
        eval_score = evaluate_board(board, maximizing_player)
        tt_store(board_hash, depth, eval_score, TT_EXACT, 0, tt_age)
        return (eval_score, None, None)

    moves_items = list(moves_dict.items())

    scores = [(item, _enhanced_move_ordering(item, depth, prev_best)) for item in moves_items]

    is_maximizer = player.lower() == maximizing_player.lower()
    if is_maximizer:

        scores.sort(key=lambda x: (x[1], compute_zobrist_hash(x[0][1])), reverse=True)
    else:

        scores.sort(key=lambda x: (x[1], -compute_zobrist_hash(x[0][1])))

    moves_items = [item for item, _ in scores]

    next_player = "White" if player.lower() == "black" else "Black"

    if is_maximizer:
        best_score = float('-inf')
        for move_key, move in moves_items:
            try:
                score, _, _ = alpha_beta_with_time_check(
                    move, depth - 1, alpha, beta, next_player, maximizing_player,
                    move_generator, time_start, time_limit,
                    (best_move_key, best_move) if best_move else None
                )

                if score > best_score:
                    best_score = score
                    best_move = move
                    best_move_key = move_key

                alpha = max(alpha, best_score)

                if beta <= alpha:
                    move_hash = compute_zobrist_hash(move)
                    move_hash_str = str(move_hash)

                    if depth not in killer_moves:
                        killer_moves[depth] = {}
                    killer_moves[depth][move_hash_str] = killer_moves[depth].get(move_hash_str, 0) + depth * depth
                    history_table[move_hash_str] = history_table.get(move_hash_str, 0) + depth * depth
                    break

            except TimeoutError:
                if best_move is None and moves_items:
                    best_move_key, best_move = moves_items[0]
                raise

        if best_move is not None:
            best_move_hash = compute_zobrist_hash(best_move)

        if best_score <= alpha:
            tt_flag = TT_UPPER
        elif best_score >= beta:
            tt_flag = TT_LOWER
        else:
            tt_flag = TT_EXACT

        tt_store(board_hash, depth, best_score, tt_flag, best_move_hash, tt_age)

    else:
        best_score = float('inf')
        for move_key, move in moves_items:
            try:
                score, _, _ = alpha_beta_with_time_check(
                    move, depth - 1, alpha, beta, next_player, maximizing_player,
                    move_generator, time_start, time_limit,
                    (best_move_key, best_move) if best_move else None
                )

                if score < best_score:
                    best_score = score
                    best_move = move
                    best_move_key = move_key

                beta = min(beta, best_score)

                if beta <= alpha:
                    move_hash = compute_zobrist_hash(move)
                    move_hash_str = str(move_hash)

                    if depth not in killer_moves:
                        killer_moves[depth] = {}
                    killer_moves[depth][move_hash_str] = killer_moves[depth].get(move_hash_str, 0) + depth * depth
                    history_table[move_hash_str] = history_table.get(move_hash_str, 0) + depth * depth
                    break

            except TimeoutError:
                if best_move is None and moves_items:
                    best_move_key, best_move = moves_items[0]
                raise

        if best_move is not None:
            best_move_hash = compute_zobrist_hash(best_move)

        if best_score <= alpha:
            tt_flag = TT_UPPER
        elif best_score >= beta:
            tt_flag = TT_LOWER
        else:
            tt_flag = TT_EXACT

        tt_store(board_hash, depth, best_score, tt_flag, best_move_hash, tt_age)

    return best_score, best_move, best_move_key

@cython.boundscheck(False)
@cython.wraparound(False)
cdef uint64_t compute_zobrist_hash_c(int * black_rows, int * black_cols, int black_count,
                                     int * white_rows, int * white_cols, int white_count):
    
    cdef uint64_t hash_value = 0
    cdef int i, idx

    for i in range(black_count):
        idx = coord_to_index_c(black_rows[i], black_cols[i])
        if idx >= 0:
            hash_value ^= ZOBRIST_TABLE_C[0][idx]

    for i in range(white_count):
        idx = coord_to_index_c(white_rows[i], white_cols[i])
        if idx >= 0:
            hash_value ^= ZOBRIST_TABLE_C[1][idx]

    return hash_value

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef uint64_t compute_zobrist_hash(list board):
    
    cdef int * black_rows = <int *> malloc(MAX_MARBLES * sizeof(int))
    cdef int * black_cols = <int *> malloc(MAX_MARBLES * sizeof(int))
    cdef int * white_rows = <int *> malloc(MAX_MARBLES * sizeof(int))
    cdef int * white_cols = <int *> malloc(MAX_MARBLES * sizeof(int))
    cdef int black_count = 0
    cdef int white_count = 0
    cdef uint64_t hash_value = 0

    if not black_rows or not black_cols or not white_rows or not white_cols:
        if black_rows: free(black_rows)
        if black_cols: free(black_cols)
        if white_rows: free(white_rows)
        if white_cols: free(white_cols)
        raise MemoryError("memory issue 1")
    try:

        copy_marbles_to_c_array(board[0], black_rows, black_cols, &black_count)
        copy_marbles_to_c_array(board[1], white_rows, white_cols, &white_count)

        hash_value = compute_zobrist_hash_c(black_rows, black_cols, black_count, white_rows, white_cols, white_count)

        return hash_value
    finally:

        if black_rows: free(black_rows)
        if black_cols: free(black_cols)
        if white_rows: free(white_rows)
        if white_cols: free(white_cols)

@cython.boundscheck(False)
@cython.wraparound(False)
def board_to_key(list board):
    
    return compute_zobrist_hash(board)

def find_best_move(list board, str player, int depth=5, double time_limit=10.0, object from_move_generator=None):
    
    cdef int min_depth = 3
    cdef double start_time, current_time, elapsed, max_search_time, depth_start_time, remaining_time
    cdef str move_str
    cdef double score, last_best_score = 0.0
    cdef int current_depth, max_reached_depth = 0
    cdef tuple prev_best, result
    cdef list last_best_move = None
    cdef object last_best_move_key = None

    if depth < min_depth:
        depth = min_depth

    if from_move_generator is None:
        try:
            from next_move_generator_cy import generate_all_next_moves
            from_move_generator = generate_all_next_moves
        except ImportError:
            raise ImportError("Move generator not provided or not found.")

    start_time = time.time()
    max_search_time = time_limit * 0.95

    clear_tt()
    killer_moves.clear()
    history_table.clear()

    for current_depth in range(min_depth, depth + 1):
        current_time = time.time()
        elapsed = current_time - start_time

        if elapsed > max_search_time:
            print(f"Time limit approaching, stopping at depth {current_depth - 1}")
            break

        print(f"\nSearching depth {current_depth}...\n")

        try:
            depth_start_time = time.time()
            remaining_time = max_search_time - elapsed

            prev_best = (last_best_move_key, last_best_move) if last_best_move else None

            score, move, move_key = alpha_beta_with_time_check(
                board, current_depth, float('-inf'), float('inf'),
                player, player, from_move_generator, depth_start_time, remaining_time,
                prev_best
            )

            if move is not None:
                last_best_move = move
                last_best_move_key = move_key
                last_best_score = score
                max_reached_depth = current_depth

                move_str = get_move_string_from_key(move_key)
                print(f"Depth {current_depth} best move: {move_str} (score: {score:.2f})")

        except TimeoutError:
            print(f"Time limit reached during depth {current_depth} search")
            break

    if last_best_move is None:
        try:
            current_time = time.time()
            remaining_time = max(time_limit * 0.1, time_limit - (current_time - start_time))
            print(f"No move found. Emergency search with {remaining_time:.4f}s")

            score, last_best_move, last_best_move_key = alpha_beta_with_time_check(
                board, 1, float('-inf'), float('inf'),
                player, player, from_move_generator, current_time, remaining_time
            )

            if last_best_move:
                last_best_score = score
                max_reached_depth = 1

        except TimeoutError:
            print("Emergency search timed out")
            color = "BLACK" if player.lower() == "black" else "WHITE"
            moves_dict = from_move_generator(board, color)
            if moves_dict:

                sorted_moves = sorted(moves_dict.items(), key=lambda x: compute_zobrist_hash(x[1]))
                last_best_move_key, last_best_move = sorted_moves[0]

    move_str = get_move_string_from_key(last_best_move_key) if last_best_move_key else "No move found"
    end_time = time.time()
    total_time = end_time - start_time

    print(f"Search completed in {total_time:.4f}s - Max depth: {max_reached_depth}, Final score: {last_best_score:.2f}")

    return last_best_move, move_str, {}, total_time

def get_move_string_from_key(move_key):
    
    if move_key is None:
        return "No move found"

    source_coords, dest_coords = move_key

    letter_map = {i: chr(ord('A') + i - 1) for i in range(1, 10)}

    from_str = ''.join(f"{letter_map[r]}{c}" for r, c in sorted(source_coords))
    to_str = ''.join(f"{letter_map[r]}{c}" for r, c in sorted(dest_coords))

    return f"{from_str},{to_str}"