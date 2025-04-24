from typing import List, Sequence

import numpy as np
from avns.local_search_operator import LocalSearchArgs, LocalSearchOperator


class LocalSearch:
    def __init__(self,
                 operators: List[LocalSearchOperator]):
        self.operators: List[LocalSearchOperator] = operators
        self.operator_available_args: List[Sequence[LocalSearchArgs]] = [[] for _ in range(len(self.operator_available_args))]

    def __call__(self, solution):
        pass
        # do until no operators feasible
        # for all operators, if its empty, try to re-fill
        # if still empty, then set this operator as unavailable
        
        