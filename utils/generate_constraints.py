import os

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
    'rule': ['rules']
}


class Tmp:
    def __init__(self, data_dir_path, file_names_dict):
        self.data_dir_path = data_dir_path
        self.file_names_dict = file_names_dict
        
        # データ
        self.L = None
        self.U = None
        self.S = None 

        # 仮
        self.KB_origin = None
        self.KB = None

    def load_data(self):
        L_path = [os.path.join(self.data_dir_path, file_name + '.csv') for file_name in self.file_names_dict['supervised']]
        U_path = [os.path.join(self.data_dir_path, file_name + '.csv') for file_name in self.file_names_dict['unsupervised']]
        L_tmp = [np.array(pd.read_csv(path, index_col=0)) for path in L_path]
        U_tmp = [np.array(pd.read_csv(path, index_col=0)) for path in U_path]
        L = np.stack([arr for arr in L_tmp])
        # U = np.stack([arr for arr in U_tmp])
        U = U_tmp[0]

        self.L = L
        self.U = U 

        # 仮実装
        S_tmp = []
        for i in range(len(L)):
            S_tmp.append(np.concatenate((L[i][:, :2], U), axis=0))
        
        S = np.stack(S_tmp)
        self.S = S

    def load_rules(self):
        rules_path = os.path.join(self.data_dir_path, self.file_names_dict['rule'][0] + '.txt')
        fol_processor = FOLProcessor(rules_path)
        self.KB_origin = fol_processor.KB
        self.KB = fol_processor.main_v2()
    
    # def 

    # predicate のマッチング
    # cp.Variable の数え上げと定義
    # objective function の生成
    # constraints の生成
    
    # ー＞ あとは cp.Problem(objective_function, constraints) で solve すれば解ける

    
    # def _construct_pointwise_constraints(self, ):


    # def _construct_logical_constraints(self, ):


    # def _construct_consistency_constraints(self, ):


    # def construct_constraints(self):
    #     pointwise = self._construct_pointwise_constraints()
    #     logical = self._construct_logical_constraints()
    #     consistency = self._construct_consistency_constraints()

    #     constraints = pointwise + logical + consistency

    #     return constraints


    # def construct_objective_function(self, c1=100, c2=100):















































