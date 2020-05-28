from pulp import *


class Tindar:
    '''Class to generate and solve Tindar pairing problems

    Input
    -----
    love_matrix: np.array
        square matrix inidacting which person is interested in which other person
    n: integer
        size of love_matrix if love_matrix not supplied
    difficulty: 1 < integer < 5
        difficulty of problem if love_matrix not supplied

    '''
    def __init__(love_matrix=None, n=None, difficulty=None):
        self.love_matrix = love_matrix
        self.n = n
        self.difficulty = difficulty
        if love_matrix is None:
            if (n is None) or (difficulty is None):
                raise ValueError("Love matrix not supplied, must supply other parameters")
        
        else:
            if (n is not None) or (difficulty is not None):
                raise Warning("Both love_matrix and other parameters supplied"
                              "Using love_matrix for computations"
                              "Overwrite with tindar_obj.create_problem()")

    def create_love_matrix(self):
        pass
