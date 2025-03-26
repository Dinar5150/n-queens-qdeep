# Copyright 2020 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("agg")    # must select backend before importing pyplot
import matplotlib.pyplot as plt

# Note: The original solver is replaced by the QDeepHybridSolver from qdeepsdk.
# Also note: the function 'exact_cover_bqm' is assumed to return a dimod.BinaryQuadraticModel.
from exact_cover import exact_cover_bqm
from qdeepsdk import QDeepHybridSolver

import dimod  # assuming exact_cover_bqm returns a dimod.BQM

NUM_READS = 1000

def build_subsets(n):
    """Returns a list of subsets of constraints corresponding to every
    position on the chessboard.

    Each constraint is represented by a unique number (ID). Each subset
    should contain:
    1) Exactly one column constraint ID (0 to n-1).
    2) Exactly one row constraint ID (n to 2*n-1).
    3) At most one diagonal (top-left to bottom-right) constraint ID (2*n to
       4*n-4).
    4) At most one anti-diagonal (bottom-left to top-right) constraint ID (4*n-3
       to 6*n-7).
    """
    subsets = []
    for x in range(n):
        for y in range(n):
            col = x
            row = y + n
            subset = {col, row}
            diag = x + y + (2 * n - 1)
            min_diag = 2 * n
            max_diag = 4 * n - 4
            if min_diag <= diag <= max_diag:
                subset.add(diag)
            anti_diag = (n - 1 - x + y) + (4 * n - 4)
            min_anti_diag = 4 * n - 3
            max_anti_diag = 6 * n - 7
            if min_anti_diag <= anti_diag <= max_anti_diag:
                subset.add(anti_diag)
            subsets.append(subset)
    return subsets

def handle_diag_constraints(bqm, subsets, diag_constraints):
    """Update bqm with diagonal (and anti-diagonal) constraints.
    Duplicates are penalized.
    """
    for constraint in diag_constraints:
        for i in range(len(subsets)):
            if constraint in subsets[i]:
                for j in range(i):
                    if constraint in subsets[j]:
                        bqm.add_interaction(i, j, 2)
    return bqm

def n_queens(n):
    """Returns a potential solution to the n-queens problem as a list of sets,
    each containing constraint IDs representing a queen's location.
    """
    num_row_col_constraints = 2 * n
    row_col_constraint_ids = set(range(num_row_col_constraints))
    num_diag_constraints = 4 * n - 6  # includes anti-diag constraints
    diag_constraint_ids = set(range(num_row_col_constraints, num_row_col_constraints + num_diag_constraints))
    
    # Build subsets of constraint IDs. Each subset will be a variable.
    subsets = build_subsets(n)
    
    # Build the BQM for the exact cover problem (row/column constraints)
    bqm = exact_cover_bqm(row_col_constraint_ids, subsets)
    
    # Add diagonal and anti-diagonal constraints (penalizing duplicates)
    bqm = handle_diag_constraints(bqm, subsets, diag_constraint_ids)
    
    # Convert the BQM to a QUBO matrix.
    # We assume bqm is a dimod.BinaryQuadraticModel and use its to_qubo() method.
    Q_dict, offset = bqm.to_qubo()
    num_vars = len(subsets)
    Q_matrix = np.zeros((num_vars, num_vars))
    for (i, j), bias in Q_dict.items():
        Q_matrix[i, j] = bias

    # Initialize the hybrid solver from qdeepsdk.
    solver = QDeepHybridSolver()
    # Set the authentication token as required (replace with your actual token)
    solver.token = "your-auth-token-here"
    # Optionally, adjust solver parameters (measurement budget, number of reads, etc.)
    solver.m_budget = 1000    # or another value as needed
    solver.num_reads = NUM_READS

    # Solve the QUBO problem
    response = solver.solve(Q_matrix)
    # The response is expected to have a structure with a key "QdeepHybridSolver" that contains
    # a "configuration" list representing the solution vector.
    solution_vector = response["QdeepHybridSolver"]["configuration"]
    
    # Return a list of subsets for which the corresponding bit is 1
    return [subsets[i] for i, bit in enumerate(solution_vector) if bit == 1.0]

def is_valid_solution(n, solution):
    """Check that solution is valid by making sure all constraints were met.

    Args:
        n: Number of queens in the problem.
        solution: A list of sets, each containing constraint IDs that represent a queen's location.
    """
    count = Counter()
    for queen in solution:
        count += Counter(queen)
    # Check row/col constraints
    for i in range(2 * n):
        if count[i] != 1:
            if i < n:
                col = i
                print("Column {} has {} queens.".format(col, count[i]))
            else:
                row = abs(i - (2 * n - 1))  # Convert constraint id to row index
                print("Row {} has {} queens.".format(row, count[i]))
            return False
    # Check diag/anti-diag constraints
    for i in range(2 * n, 4 * n - 6):
        if count[i] > 1:
            if i <= 4 * n - 4:
                print("Top-left to bottom-right diagonal {} has {} queens.".format(i, count[i]))
            else:
                print("Bottom-left to top-right diagonal {} has {} queens.".format(i, count[i]))
            return False
    return True

def plot_chessboard(n, queens):
    """Create a chessboard with queens using matplotlib.
    Image is saved in the root directory.
    Returns the image file name.
    """
    chessboard = np.zeros((n, n))
    chessboard[1::2, 0::2] = 1
    chessboard[0::2, 1::2] = 1

    # Adjust fontsize for readability
    if n <= 10:
        fontsize = 30
    elif n <= 20:
        fontsize = 10
    else:
        fontsize = 5

    plt.xticks(np.arange(n))
    plt.yticks(np.arange(n))
    plt.imshow(chessboard, cmap='binary')

    # Place queens on the board
    for subset in queens:
        x = y = -1
        for constraint in subset:
            if constraint < n:
                x = constraint
            elif n <= constraint < 2 * n:
                y = abs(constraint - (2 * n - 1))  # Convert constraint ID to row index
        if x != -1 and y != -1:
            plt.text(x, y, u"\u2655", fontsize=fontsize, ha='center',
                     va='center', color='black' if (x - y) % 2 == 0 else 'white')

    file_name = "{}-queens-solution.png".format(n)
    plt.savefig(file_name)
    return file_name

def get_sanitized_input():
    while True:
        print("Enter the number of queens to place (n > 0):")
        n = input()
        try:
            n = int(n)
            if n <= 0:
                print("Input must be greater than 0.")
                continue
            if n >= 200:
                print("Problems with large n will run very slowly.")
        except ValueError:
            print("Input type must be int.")
            continue
        return n

if __name__ == "__main__":
    n = get_sanitized_input()
    if n > 20:
        print("Solution image is large and may be difficult to view.")
        print("Plot settings in plot_chessboard() may need adjusting.")
    print("Trying to place {n} queens on a {n}*{n} chessboard.".format(n=n))
    solution = n_queens(n)
    if is_valid_solution(n, solution):
        print("Solution is valid.")
    else:
        print("Solution is invalid.")
    file_name = plot_chessboard(n, solution)
    print("Chessboard created. See: {}".format(file_name))
