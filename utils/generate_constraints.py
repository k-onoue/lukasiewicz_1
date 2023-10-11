import cvxpy as cp
import numpy as np
import pandas as pd

from .operators import negation
from .operators import weak_conjunction, strong_disjunction
from .operators import weak_disjunction, weak_conjunction
from .operators import implication

from .misc import count_neg, get_first_neg_index
from .misc import Predicate

from .process_fol import FOLProcessor


symbols_1 = ['¬', '∧', '∨', '⊗', '⊕', '→']

file_names_dict = {
    'supervised': ['L1', 'L2', 'L3'],
    'unsupervised': ['U'],
    'rule': ['rules_2']
}




# どうせ，cp.Variable の生成も伴うので，objective_function の生成も同時にできるようにしたい
# もしくは objective_function の生成クラスと，
# それら２つをラップするようなものも作って 
# 先に knowledgebase を load するようにし，cp.Variable を共有する
class ConstraintsGenerator:
    def __init__(self, data_dir_path, file_names_dict):
        self.data_dir_path = data_dir_path
        self.file_names_dict = file_names_dict

        self.KB = None


    
    def _load_data(self):
        L = 
        U = 
        S = 


    def _construct_pointwise_constraints(self, ):

    
    def _construct_logical_constraints(self, ):


    def _construct_consistency_constraints(self, ):


    def construct_constraints(self):
        pointwise = self._construct_pointwise_constraints()
        logical = self._construct_logical_constraints()
        consistency = self._construct_consistency_constraints()

        constraints = pointwise + logical + consistency

        return constraints














































