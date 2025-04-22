from problem.hvrp3l import HVRP3L
from problem.solution import Solution
class AVNS:
    def __init__(self):
        pass
    
    def solve(self, problem:HVRP3L)->Solution:
        solution: Solution = Solution(problem)