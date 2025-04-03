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
matplotlib.use("agg")    # select backend before importing pyplot
import matplotlib.pyplot as plt

# Note: This code assumes that exact_cover.exact_cover_bqm returns a dimod.BinaryQuadraticModel
# and that qdeepsdk.QDeepHybridSolver is available for solving the QUBO.
from exact_cover import exact_cover_bqm
from qdeepsdk import QDeepHybridSolver

def build_subsets(n):
    """
    Returns a list of subsets of constraint IDs for every board position.
    Each subset represents the constraints satisfied by placing a queen on that square.
    Constraints:
      - Column: IDs 0 to n-1 (x coordinate)
      - Row: IDs n to 2*n-1 (y coordinate)
      - Diagonal: IDs 2*n to 3*n+... (we use a unique id for each diagonal)
      - Anti-diagonal: following the diagonals.
    (The specific IDs here are chosen to ensure they are unique.)
    """
    subsets = []
    for x in range(n):
        for y in range(n):
            subset = set()
            # Column constraint: ID = x
            subset.add(x)
            # Row constraint: ID = n + y
            subset.add(n + y)
            # Main diagonal constraint: using x-y offset shifted by n-1 so it’s nonnegative
            diag = x - y + (n - 1)
            subset.add(2 * n + diag)
            # Anti-diagonal constraint: using x+y
            anti_diag = x + y
            subset.add(3 * n + (n - 1) + anti_diag)
            subsets.append(subset)
    return subsets

def handle_diag_constraints(bqm, subsets, diag_constraints):
    """
    Update the BQM with additional diagonal constraints.
    For each diagonal (or anti-diagonal) constraint, add an interaction between every
    pair of positions (i, j) that share that constraint.
    """
    for constraint in diag_constraints:
        for i in range(len(subsets)):
            if constraint in subsets[i]:
                for j in range(i):
                    if constraint in subsets[j]:
                        bqm.add_interaction(i, j, 2)
    return bqm

def n_queens(n):
    """
    Constructs and solves the n-queens problem as an exact cover problem.
    Returns a list of subsets corresponding to board positions chosen (with bit==1).
    """
    # Row and column constraints: total 2*n constraints.
    num_row_col_constraints = 2 * n
    row_col_constraint_ids = set(range(num_row_col_constraints))
    
    # Diagonal and anti-diagonal constraints.
    # For our purposes we consider all diagonals (each square belongs to one main diagonal
    # and one anti-diagonal). We assume an approximate count.
    num_diag_constraints = (2 * n - 1) * 2
    diag_constraint_ids = set(range(num_row_col_constraints, num_row_col_constraints + num_diag_constraints))
    
    # Build the subsets representing all board positions.
    subsets = build_subsets(n)
    
    # Build the BQM for the exact cover problem (row and column constraints).
    bqm = exact_cover_bqm(row_col_constraint_ids, subsets)
    
    # Add diagonal (and anti-diagonal) constraints to penalize conflicts.
    bqm = handle_diag_constraints(bqm, subsets, diag_constraint_ids)
    
    # Convert the BQM to a QUBO matrix.
    Q_dict, offset = bqm.to_qubo()
    num_vars = len(subsets)
    Q_matrix = np.zeros((num_vars, num_vars))
    for (i, j), bias in Q_dict.items():
        Q_matrix[i, j] = bias

    # Initialize the hybrid solver.
    solver = QDeepHybridSolver()
    solver.token = "your-auth-token"  # Replace with your actual token if needed.
    solver.m_budget = 1000
    solver.num_reads = 100

    # Solve the QUBO problem.
    response = solver.solve(Q_matrix)
    solution_vector = response["QdeepHybridSolver"]["configuration"]
    
    # Return the list of subsets corresponding to positions where the bit is 1.
    return [subsets[i] for i, bit in enumerate(solution_vector) if bit == 1.0]

def is_valid_solution(n, solution):
    """
    Checks that the solution satisfies:
      - Exactly one queen per row and per column.
      - No two queens share a diagonal.
    """
    count = Counter()
    for queen in solution:
        count += Counter(queen)
    # Check row and column constraints.
    for i in range(2 * n):
        if count[i] != 1:
            if i < n:
                print("Column {} has {} queens.".format(i, count[i]))
            else:
                print("Row {} has {} queens.".format(i - n, count[i]))
            return False
    # Check diagonal constraints: no diagonal may have more than one queen.
    for i in range(2 * n, 2 * n + (2 * n - 1) * 2):
        if count[i] > 1:
            print("Diagonal constraint {} violated with {} queens.".format(i, count[i]))
            return False
    return True

def plot_chessboard(n, queens):
    """
    Creates an image of an n x n chessboard and plots the queens using matplotlib.
    The image is saved to a file and the file name is returned.
    """
    chessboard = np.zeros((n, n))
    # Create a checkerboard pattern.
    chessboard[1::2, 0::2] = 1
    chessboard[0::2, 1::2] = 1

    # Adjust font size based on board size.
    if n <= 10:
        fontsize = 30
    elif n <= 20:
        fontsize = 10
    else:
        fontsize = 5

    plt.xticks(np.arange(n))
    plt.yticks(np.arange(n))
    plt.imshow(chessboard, cmap='binary')

    # Place queens on the board.
    for subset in queens:
        x = y = -1
        for constraint in subset:
            if constraint < n:
                x = constraint
            elif n <= constraint < 2 * n:
                # Convert row constraint ID to row index.
                y = (2 * n - 1) - constraint
        if x != -1 and y != -1:
            plt.text(x, y, u"\u2655", fontsize=fontsize, ha='center',
                     va='center', color='black' if (x - y) % 2 == 0 else 'white')

    file_name = "{}-queens-solution.png".format(n)
    plt.savefig(file_name)
    return file_name

def get_sanitized_input():
    """
    Prompts the user to enter a positive integer.
    For the n-queens problem, n must be 1 or at least 4.
    """
    while True:
        print("Enter the number of queens to place (n > 0):")
        n_str = input()
        try:
            n = int(n_str)
            if n <= 0:
                print("Input must be greater than 0.")
                continue
            if n == 2 or n == 3:
                print("No solution exists for n = {} (n must be 1 or >= 4).".format(n))
                continue
            if n >= 200:
                print("Problems with large n will run very slowly.")
            return n
        except ValueError:
            print("Input type must be an integer.")
            continue

if __name__ == "__main__":
    n = get_sanitized_input()
    if n > 20:
        print("Solution image is large and may be difficult to view.")
        print("Plot settings in plot_chessboard() may need adjusting.")
    print("Trying to place {n} queens on a {n}x{n} chessboard.".format(n=n))
    solution = n_queens(n)
    if is_valid_solution(n, solution):
        print("Solution is valid.")
    else:
        print("Solution is invalid.")
    file_name = plot_chessboard(n, solution)
    print("Chessboard created. See: {}".format(file_name))
