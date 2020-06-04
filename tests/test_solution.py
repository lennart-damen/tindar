import sys
from pathlib import Path
import pytest
import itertools

PROJECT_DIR = str(Path(__file__).resolve().parents[1])
sys.path.insert(1, PROJECT_DIR+"/src")

from tindar import Tindar, TindarGenerator
from timer import Timer

# Test data
n_list = [10, 30, 100, 1000]
connectedness_list = [1, 3, 8]

tindar_problems = [
    TindarGenerator(tup[0], tup[1])
    for tup in zip(n_list, connectedness_list)
]

tindars = [
    Tindar(tindar_problem=tindar_problem)
    for tindar_problem in tindar_problems
]


def _aux_solution_tester(solution):
    m, n = solution.shape

    assert m == n, "solution is not square"

    assert (solution.sum(axis=1) <= 1).all(), "solution assigns one person to two other people"
    assert (solution.sum(axis=0) <= 1).all(), "solution assigns one person to two other people"
    for i in range(n):
        for j in range(i+1, n):
            assert solution[i, j] == solution[i, j], "solution is not symmetric"


def test_heuristic_solution():
    for tindar in tindars:
        with Timer():
            tindar.solve_problem(kind="heuristic")
        _aux_solution_tester(tindar.solution_vars(kind="heuristic", verbose=False))

# def test_pulp_solution():
#     for tindar in tindars:
#         tindar.create_problem()
#         with Timer():
#             tindar.solve_problem()
#         pulp_solution = tindar.solution_obj()
#         numpy_solution = "blabla"  # TODO: replace by method call

#         pass
