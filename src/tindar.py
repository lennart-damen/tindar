# Tindar class version 0:
# copied the jupyter notebook
# grouped cells into functions
# converted global variables to object
# attributes by adding self.___ where appropriate

from pulp import *
import numpy as np
from pathlib import Path

PROJECT_DIR = str(Path(__file__).resolve().parents[1])


class Tindar:
    '''Class to solve Tindar pairing problems

    Input
    -----
    love_matrix: np.array
        square matrix indicating which person is interested
        in which other person
    '''

    def __init__(self, love_matrix):
        self.check_init(love_matrix)

        self.love_matrix = love_matrix
        self.n = love_matrix.shape[0]
        self.x_names = [f"x_{i}_{j}" for i in range(n) for j in range(n)]
        self.x = [LpVariable(name=x_name, cat="Binary")
                  for x_name in self.x_names]
        self.x_np = np.array(self.x).reshape((n, n))

    @staticmethod
    def check_init(love_matrix):
        # type check
        if not isinstance(love_matrix, np.ndarray):
            raise ValueError("love_matrix is not a numpy array")

        # shape check
        m, n = love_matrix.shape
        if m != n:
            raise ValueError(f"love_matrix is not square: love_matrix.shape"
                             f"= {love_matrix.shape}")

        # diagonal zero check
        for i in range(n):
            if love_matrix[i, i] != 0:
                raise ValueError("love_matrix diagonal contains nonzeros")

    # Symmetry constraints: if one is paired, the other is paired
    def create_symmetry_constraints(self, inplace=True):
        tups = [(i, j) for i in range(self.n) for j in range(i+1, self.n)]

        # Left-hand side
        lhs_symmetry = [
            LpAffineExpression(
                [(self.x_np[tup[0], tup[1]], 1), (self.x_np[tup[1], tup[0]], -1)],
                name=f"lhs_sym_{tup[0]}_{tup[1]}"
            )
            for tup in tups
        ]

        # Constraints
        constraints_symmetry = [
            LpConstraint(
                e=lhs_s,
                sense=0,
                name=f"constraint_sym_{tups[i][0]}_{tups[i][1]}",
                rhs=0
            )
            for i, lhs_s in enumerate(lhs_symmetry)
        ]

        # Verification
        if len(constraints_symmetry) != (self.n**2-self.n)/2:
            raise Exception(
                "Symmetry constraints not constructed right:"
                f"love_matrix.shape = {self.love_matrix.shape},"
                f"len(constraints_symmetry) should be {(self.n**2-self.n)/2}"
                f", actually is {len(constraints_symmetry)}"
            )

        # Function behaviour
        if inplace:  # object is modified, no return value
            self.constraints_symmetry = constraints_symmetry
        else:  # only result is returned
            return constraints_symmetry

    # Feasibility constraints: only pairs if person likes the other
    def create_like_constraints(self, inplace=True):
        tups = [(i, j) for i in range(self.n) for j in range(self.n)]

        # Left-hand side
        lhs_like = [
            LpAffineExpression(
                [(self.x_np[tup[0], tup[1]], 1)],
                name=f"lhs_like_{tup[0]}_{tup[1]}"
            )
            for tup in tups
        ]

        # Constraints
        constraints_like = [
            LpConstraint(
                e=lhs_l,
                sense=-1,
                name=f"constraint_like_{tups[i][0]}_{tups[i][1]}",
                rhs=self.love_matrix[tups[i][0], tups[i][1]]
            )
            for i, lhs_l in enumerate(lhs_like)
        ]

        # Verification
        if len(constraints_like) != self.n**2:
            raise Exception(
                "Liking constraints not constructed right:"
                f"A.shape = {self.love_matrix.shape}, len(constraints_like)"
                f"should be {self.n**2}, actually is {len(constraints_like)}"
            )

        # Function behaviour
        if inplace:  # object is modified, no return value
            self.constraints_like = constraints_like
        else:  # only result is returned
            return constraints_like

    # Single assignment: one person can have at most one other person
    def create_single_assignment_constraints(self, inplace=True):
        # Left-hand side: rowsum <= 1
        lhs_single_rowsum = [
            LpAffineExpression(
                [(self.x_np[i, j], 1) for j in range(self.n)],
                name=f"lhs_single_rowsum_{i}"
            )
            for i in range(self.n)
        ]

        # Left-hand side: colsum <= 1
        lhs_single_colsum = [
            LpAffineExpression(
                [(self.x_np[i, j], 1) for i in range(self.n)],
                name=f"lhs_single_colsum_{j}"
            )
            for j in range(self.n)
        ]

        # Constraints
        constraints_single_rowsum = self.make_single_constraints(
            lhs_single_rowsum, "rowsum")
        constraints_single_colsum = self.make_single_constraints(
            lhs_single_colsum, "colsum")

        # Verification
        self.check_single_constraints(constraints_single_rowsum, "rowsum")
        self.check_single_constraints(constraints_single_colsum, "colsum")

        # Function behaviour
        if inplace:  # object is modified, no return value
            self.constraints_single_rowsum = constraints_single_rowsum
            self.constraints_single_colsum = constraints_single_colsum

        else:  # only result is returned
            return constraints_single_rowsum, constraints_single_colsum

    # Auxiliary functions for single assigment constraints
    @staticmethod
    def make_single_constraints(lhs_single, kind):
        constraints_single = [
            LpConstraint(
                e=lhs_s,
                sense=-1,
                name=f"constraint_single_{kind}_{i}",
                rhs=1
            )
            for i, lhs_s in enumerate(lhs_single)
        ]

        return constraints_single

    def check_single_constraints(self, constraints_single, kind):
        if len(constraints_single) != self.n:
            raise Exception(
                f"Constraints single {kind} not constructed right:"
                f"A.shape = {self.love_matrix.shape}, "
                f"len(constraints_single_{kind}) should be {self.n}, "
                f"actually is {len(constraints_single)}"
            )

    def create_all_constraints(self):
        self.create_symmetry_constraints()
        self.create_like_constraints()
        self.create_single_assignment_constraints()

        self.constraints_all = [
            *self.constraints_symmetry,
            *self.constraints_like,
            *self.constraints_single_rowsum,
            *self.constraints_single_colsum
        ]

    def create_problem(self):
        # Initialize constraints and objective
        self.create_all_constraints()
        self.objective = LpAffineExpression([(x_i, 1) for x_i in self.x])

        # Create PuLP problem
        self.prob_pulp = LpProblem("The_Tindar_Problem", LpMaximize)
        self.prob_pulp += self.objective

        for c in self.constraints_all:
            self.prob_pulp += c

    def write_problem(self, path=PROJECT_DIR+"/models/Tindar.lp"):
        self.prob_pulp.writeLP(path)

    def solve_problem(self):
        self.prob_pulp.solve()

    def solution_status(self, verbose=True):
        stat = LpStatus[self.prob_pulp.status]
        if verbose:
            print("Status:", stat)
        return stat

    def solution_vars(self, verbose=True):
        vars_pulp = self.prob_pulp.variables()
        if verbose:
            for v in vars_pulp:
                print(v.name, "=", v.varValue)
        return vars_pulp

    def solution_obj(self, verbose=True):
        obj = value(self.prob_pulp.objective)
        if verbose:
            print("Number of lovers connected = ", obj)
        return obj


class TindarGenerator:
    '''Class to generate Tindar objects randomly
    n: integer
        number of people in the model
    connectedness: 1 < integer < 10
        connectedness of the Tindar problem for humans,
        implemented as bernouilli probability for edges
        to be generated
    '''
    MIN_CONNECTEDNESS = 1
    MAX_CONNECTEDNESS = 10
    MIN_EDGE_PROB = 0.05
    MAX_EDGE_PROB = 0.75

    def __init__(self, n, connectedness):
        self.check_init(n, connectedness)
        self.n = n
        self.connectedness = connectedness
        self.create_love_matrix()

    # Input validation
    @classmethod
    def check_init(self, n, connectedness):
        # n
        if not isinstance(n, int):
            raise ValueError(f"TindarGenerator init error: "
                             f"type(n) = {type(n)}")
        if n <= 0:
            raise ValueError(f"TindarGenerator init error: "
                             f"n={n} < 0")

        # connectedness
        if not isinstance(connectedness, int):
            raise ValueError(f"TindarGenerator init error: "
                             f"type(connectedness) = {type(connectedness)}")
        if not (self.MIN_CONNECTEDNESS <= connectedness <= self.MAX_CONNECTEDNESS):
            raise ValueError(f"TindarGenerator init error: "
                             f"connectedness={connectedness} not between 1 and 10")

    @classmethod
    def bernouilli_parameter(self, connectedness):
        diff_scaled = (connectedness-self.MIN_EDGE_PROB)/self.MAX_CONNECTEDNESS
        return (diff_scaled*self.MAX_EDGE_PROB) + self.MIN_EDGE_PROB

    def create_love_matrix(self, n=None, connectedness=None, inplace=True):
        if n is None:
            n = self.n
        if connectedness is None:
            connectedness = self.connectedness

        self.p = self.bernouilli_parameter(connectedness)
        love_matrix = np.random.binomial(1, self.p, size=(n, n))

        for i in range(n):
            love_matrix[i, i] = 0

        if inplace:
            self.love_matrix = love_matrix
        else:
            return love_matrix


if __name__ == "__main__":
    n = 100
    connectedness = 1

    tindar_problem = TindarGenerator(n, connectedness)
    tindar = Tindar(tindar_problem.love_matrix)

    print(f"love_matrix:\n{tindar.love_matrix}")
    print(f"p: {tindar_problem.p}")

    tindar.create_problem()
    tindar.write_problem()
    tindar.solve_problem()

    tindar.solution_status()
    tindar.solution_obj()
    # tindar.inspect_solution_vars()
