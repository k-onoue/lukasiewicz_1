import os

import cvxpy as cp
import numpy as np
import pandas as pd

from .operators import negation
# from .operators import weak_conjunction, strong_disjunction
# from .operators import weak_disjunction, weak_conjunction
# from .operators import implication
from .operators import Semantisize_symbols

# from .misc import count_neg, get_first_neg_index
from .misc import process_neg
from .misc import Predicate
from .misc import count_an_operator, get_first_an_oprator_index

from .process_fol import FOLConverter


# symbols_1 = ['¬', '∧', '∨', '⊗', '⊕', '→']
# symbols_2 = ['∀', '∃']
# symbols_3 = ['+', '-']
# symbols = symbols_1 + symbols_2 + symbols_3

symbols_tmp = Semantisize_symbols()
symbols_1_semanticized = symbols_tmp.symbols_1_semanticized
symbols_3_semanticized = symbols_tmp.symbols_3_semanticized
symbols = list(symbols_1_semanticized.keys()) + list(symbols_3_semanticized.keys())



file_names_dict = {
    'supervised': ['L1', 'L2', 'L3'],
    'unsupervised': ['U'],
    'rule': ['rules']
}


class Setup:
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

        self.predicates_dict = None

        # ループ用
        self.len_j = None
        self.len_l = None
        self.len_h = None

        self.len_s = None

        # cvxpy.Variable
        self.w_j = None
        self.xi_jl = None
        self.xi_h = None

        # obj func
        self.objective_function = None

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

        self.len_l = len(L)

        # 仮実装
        S_tmp = []
        for i in range(self.len_l):
            S_tmp.append(np.concatenate((L[i][:, :2], U), axis=0))
        
        S = np.stack(S_tmp)
        self.S = S
        
        # 仮
        self.len_s = len(self.S[0])

    def load_rules(self):
        rules_path = os.path.join(self.data_dir_path, self.file_names_dict['rule'][0] + '.txt')
        fol_processor = FOLConverter(rules_path)
        self.KB_origin = fol_processor.KB
        self.KB = fol_processor.main_v2()

        # 仮
        self.len_h = len(self.KB) * 2

    def _define_cvxpy_variables(self):
        self.w_j = cp.Variable(shape=(self.len_j, 3))
        self.xi_jl = cp.Variable(shape=(self.len_j, self.len_l), nonneg=True)
        self.xi_h = cp.Variable(shape=(self.len_h, 1), nonneg=True)
    
    def _identify_predicates(self, KB):
        predicates = []
        for formula in KB:
            for item in formula:
                if item not in symbols and item not in predicates:
                    predicates.append(item)

        self.len_j = len(predicates)
        self._define_cvxpy_variables()
        self.predicates_dict = {predicate: Predicate(self.w_j[j, :]) for j, predicate in enumerate(predicates)}

    def identify_predicates(self):
        self._identify_predicates(self.KB_origin)

    def _calc_KB_at_datum(self, KB, datum):
        new_KB = KB
        for formula in new_KB:
            for j, item in enumerate(formula):
                if item in self.predicates_dict:
                    formula[j] = self.predicates_dict[item](datum)
            
            process_neg(formula)

            iter_num = count_an_operator(formula, '+')
            for _ in range(iter_num):
                target_idx = get_first_an_oprator_index(formula, '+')

                formula[target_idx - 1] = formula[target_idx - 1] + formula[target_idx + 1]
                formula.pop(target_idx)
                formula.pop(target_idx)

            iter_num = count_an_operator(formula, '-')
            for _ in range(iter_num):
                target_idx = get_first_an_oprator_index(formula, '-')
                formula[target_idx - 1] = formula[target_idx - 1] - formula[target_idx + 1]
                formula.pop(target_idx)
                formula.pop(target_idx)
        
        return new_KB

    def construct_objective_function(self, c1=100, c2=100):
        function = 0

        for j in range(self.len_j):
            w = self.w_j[j]
            function += 1/2 * (cp.norm2(w) ** 2)

        for j in range(self.len_j):
            for l in range(self.len_l):
                xi = self.xi_jl[j, l]
                function += c1 * xi

        for h in range(self.len_h):
            xi = self.xi_h[h, 0]
            function += c2 * xi

        self.objective_function = cp.Minimize(function)

        return self.objective_function

    def _construct_pointwise_constraints(self):
        constraints_tmp = []
        # for j in range(self.len_j):
            # w = self.w_j[j]
            # p = Predicate(w)
        
        for j, p in enumerate(self.predicates_dict.values()):
            for l in range(self.len_l):
                x = self.L[j][l, :2]
                y = self.L[j][l, 2]

                xi = self.xi_jl[j, l]

                constraints_tmp += [
                    y * (2 * p(x) - 1) >= 1 - 2 * xi
                ]
        
        return constraints_tmp
    
    def _construct_logical_constraints(self):
        KB_new = []
        for formula in self.KB:
            while '∧' in formula:
                target_idx = formula.index('∧')
                formula_1, formula_2 = formula[:target_idx], formula[target_idx + 1:]
                KB_new.append(formula_1)
                formula = formula_2
            
            KB_new.append(formula)

        # for formula in KB_tmp:
        #     for j, item in enumerate(formula):
        #         if item in self.predicates_dict:
        #             formula[j] = self.predicates_dict[item]

        constraints_tmp = []

        for u in self.U:
            KB_tmp = self._calc_KB_at_datum(KB_new, u)

            for h, formula in enumerate(KB_tmp):
                xi = self.xi_h[h, 0]

                if len(formula) == 1:
                    formula = formula[0]
                else:
                    print('There is something wrong!')
                    break

                constraints_tmp += [
                    negation(formula) <= xi
                ]
    
        return constraints_tmp

    def _construct_consistency_constraints(self):
        constraints_tmp = []
        for j, p in enumerate(self.predicates_dict.values()):
            for s in range(self.len_s):
                x = self.S[j][s]

                constraints_tmp += [
                    p(x) >= 0,
                    p(x) <= 1
                ]
        
        return constraints_tmp

    def construct_constraints(self):
        pointwise = self._construct_pointwise_constraints()
        logical = self._construct_logical_constraints()
        consistency = self._construct_consistency_constraints()

        constraints = pointwise + logical + consistency

        return constraints

    def main(self):
        self.load_data()
        self.load_rules()
        # self._identify_predicates(self.KB_origin)
        self.identify_predicates()

        obj_func = self.construct_objective_function()
        constraints = self.construct_constraints()

        return obj_func, constraints

















































