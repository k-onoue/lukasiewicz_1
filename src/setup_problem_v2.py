import os
import time

import cvxpy as cp
import numpy as np
import pandas as pd
import sympy as sp

from .operators import negation
from .operators import Semantisize_symbols

from .misc import process_neg, Predicate, is_symbol

from .process_fol_v2 import FOLConverter


# symbols_1 = ['¬', '∧', '∨', '⊗', '⊕', '→']
# symbols_2 = ['∀', '∃']
# symbols_3 = ['+', '-']
# symbols = symbols_1 + symbols_2 + symbols_3

symbols_tmp = Semantisize_symbols()
symbols_1_semanticized = symbols_tmp.symbols_1_semanticized
symbols_3_semanticized = symbols_tmp.symbols_3_semanticized
symbols = list(symbols_1_semanticized.keys()) + list(symbols_3_semanticized.keys())



# file_names_dict = {
#     'supervised': ['L1', 'L2', 'L3'],
#     'unsupervised': ['U'],
#     'rule': ['rules']
# }


class Setup:
    """
    cvxpy.Problem に渡す objective function と constraints
    の生成

    インスタンス化の際に以下の 2 つを引数として渡す
    
    data_dir_path = "./inputs/toy_data"

    file_names_dict = {
        'supervised': ['L1', 'L2', 'L3'],
        'unsupervised': ['U'],
        'rule': ['rules']
    }
    """

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

        self.KB_tmp = None

        self.predicates_dict_tmp = None
        self.predicates_dict = None

        # ループ用
        self.len_j = None
        self.len_l = None
        self.len_h = None

        self.len_s = None

        self.len_u = None
        self.len_I_h = None

        # cvxpy.Variable
        self.w_j = None

        self.xi_jl = None
        self.xi_h = None

        self.mu_jl = None
        self.mu_h = None

        self.lambda_jl = None
        self.lambda_hi = None

        self.eta_js = None
        self.eta_hat_js = None

        # coefficients of affine functions
        self.M = None 
        self.q = None

        # evaluation of p for all possible groundings
        self.p_bar = None

        # obj func
        self.objective_function = None


    def load_data(self):
        """
        .csv ファイルからデータを読み込んで，
        辞書を用いて，predicate 名でラベル付けをした
        ndarray として格納する

        {
        'p1': np.array(),
        'p2': np.array(),
        ...
        'pm': np.array()
        }
        """

        # 教師ありデータ
        self.L = {}
        for file_name in self.file_names_dict['supervised']:
            path = os.path.join(self.data_dir_path, file_name + '.csv')
            self.L[file_name[2:]] = np.array(pd.read_csv(path, index_col=0))

        # self.U = {}
        # for file_name in self.file_names_dict['unsupervised']:
        #     path = os.path.join(self.data_dir_path, file_name + '.csv')
        #     self.U[file_name] = np.array(pd.read_csv(path, index_col=0))

        # 教師なしデータ
        self.U = {}
        for file_name in self.file_names_dict['supervised']:
            path = os.path.join(self.data_dir_path, 'U.csv')
            self.U[file_name[2:]] = np.array(pd.read_csv(path, index_col=0))

        # Consistency constraints 用に上の教師ありデータと
        # 教師なしデータを合わせたもの
        self.S = {}
        for key in self.L.keys():
            self.S[key] = np.concatenate((self.L[key][:, :-1] ,self.U[key]), axis=0)


        self.len_j = len(self.L)

        # 仮
        L_tmp = next(iter(self.L.values()))
        self.len_l = len(L_tmp)
        self.dim_x_L = len(L_tmp[0, :-1]) + 1

        U_tmp = next(iter(self.U.values()))
        self.len_u = len(U_tmp)

        self.len_I_h = 2 * self.len_u

        S_tmp = next(iter(self.S.values()))
        self.len_s = len(S_tmp)


    def load_rules(self):
        """
        .txt ファイルとして保存されている Knowledge Base (KB) を読み込み，
        リストとして保持する
        """

        rules_path = os.path.join(self.data_dir_path, self.file_names_dict['rule'][0] + '.txt')
        fol_processor = FOLConverter(rules_path)
        fol_processor.main()

        self.KB_origin = fol_processor.KB_origin
        self.KB = fol_processor.KB
        self.KB_tmp = fol_processor.KB_tmp
        self.predicates_dict_tmp = fol_processor.predicates_dict_tmp

        self.len_h = len(self.KB)


    def get_M_and_q(self):
      
        predicates_sympy = list(self.predicates_dict_tmp.values())

        tmp_M = []
        tmp_q = []

        for phi_h in self.KB_tmp:

            base_M_h = np.zeros((len(phi_h), self.len_j))
            base_q_h = np.zeros((len(phi_h), 1))
            for i, formula in enumerate(phi_h):
                for j, predicate in enumerate(predicates_sympy):

                    # negation
                    val = sp.Symbol('1') - formula[0]
                    coefficient = val.coeff(predicate)
                    base_M_h[i, j] = coefficient
                
                # negation
                val = sp.Symbol('1') - formula[0]
                base_q_h[i] = val.coeff(sp.Symbol('1'))

            tmp_M_h = []
            for i in range(self.len_j):
                column = base_M_h[:, i]
                zeros = np.zeros((len(phi_h), self.len_u - 1))
                concatenated_column = np.concatenate((column[:, np.newaxis], zeros), axis=1)
                tmp_M_h.append(concatenated_column)

            M_h = np.concatenate(tmp_M_h, axis=1)

            tmp_M_h = [M_h]
            shifted_M_h = M_h
            for i in range(self.len_u - 1):
                shifted_M_h = np.roll(shifted_M_h, 1, axis=1)
                tmp_M_h.append(shifted_M_h)
            
            M_h = np.concatenate(tmp_M_h, axis=0)
            tmp_M.append(M_h)
            
            tmp_q_h = [base_q_h for u in range(self.len_u)]
            q_h = np.concatenate(tmp_q_h, axis=0)
            tmp_q.append(q_h)
        
        self.M = np.array(tmp_M)
        self.q = np.array(tmp_q)


    def _define_cvxpy_variables(self):
        self.w_j = cp.Variable(shape=(self.len_j, self.dim_x_L))

        self.xi_jl = cp.Variable(shape=(self.len_j, self.len_l), nonneg=True)
        self.xi_h = cp.Variable(shape=(self.len_h, 1), nonneg=True)
        
        self.mu_jl = cp.Variable(shape=(self.len_j, self.len_l), nonneg=True)
        self.mu_h = cp.Variable(shape=(self.len_h, 1), nonneg=True)

        self.lambda_jl = cp.Variable(shape=(self.len_j, self.len_l), nonneg=True)
        self.lambda_hi = cp.Variable(shape=(self.len_h, self.len_I_h), nonneg=True)

        self.eta_js = cp.Variable(shape=(self.len_j, self.len_s), nonneg=True)
        self.eta_hat_js = cp.Variable(shape=(self.len_j, self.len_s), nonneg=True)


    def construct_predicates_with_cvxpy(self):
        """
        KB の中の全 predicate を取得して，辞書に格納．
        predicate function を作成して対応させる
        """
        predicates = self.predicates_dict_tmp.keys()

        self._define_cvxpy_variables()

        self.predicates_dict = {predicate: Predicate(self.w_j[j]) for j, predicate in enumerate(predicates)}
        

    def evaluate_predicates_for_logical_constraints(self):

        p_bar_tmp = []
        for p_name, p in self.predicates_dict.items():
            U = self.U[p_name]
            for u in U:
                val = p(u)
                p_bar_tmp.append(val)

        self.p_bar = np.array(p_bar_tmp).reshape(-1, 1)


    def _calc_KB_at_datum(self, KB, datum):
        """
        logical constraints を構成する際に使用．
        KB の中のすべての predicate をあるデータ点で計算する
        """

        KB_new = []

        for formula in KB:
            new_formula = []

            for j, item in enumerate(formula):
                if item in self.predicates_dict:
                    new_formula.append(self.predicates_dict[item](datum))
                else:
                    new_formula.append(item)

            process_neg(new_formula)
            KB_new.append(new_formula)

        return KB_new

    
    def construct_objective_function(self, c1, c2):
        """
        目的関数を構成する．
        c1 は logical constraints を，
        c2 は consistency constraints を
        満足する度合いを表す．
        """

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

        for j in range(self.len_j):
            for l in range(self.len_l):
                mu = self.mu_jl[j, l]
                xi = self.xi_jl[j, l]
                function -= mu * xi

        for j, (p_name, p) in enumerate(self.predicates_dict.items()):
            for l in range(self.len_l):
                x = self.L[p_name][l, :-1]
                y = self.L[p_name][l, -1]

                lmbda = self.lambda_jl[j, l]
                xi = self.xi_jl[j, l]

                val = lmbda * (y * (2 * p(x) - 1) - 1 + 2 * xi)
                function -= val

        for h in range(self.len_h):
            for i in range(self.len_I_h):
                lmbda = self.lambda_hi[h, i]
                xi = self.xi_h[h, 0]

                M = self.M[h, i, :]
                q = self.q[h, i, 0]
                p_vals = self.p_bar

                M_dot_p = (M @ p_vals)[0]
            
                val = lmbda * (xi - M_dot_p - q)
                function -= val
        
        for h in range(self.len_h):
            mu = self.mu_h[h, 0]
            xi = self.xi_h[h, 0]
            function -= mu * xi

        for j, (p_name, p) in enumerate(self.predicates_dict.items()):
            for s in range(self.len_s):
                eta = self.eta_js[j, s]
                x = self.S[p_name][s]
                function -= eta * p(x)
        
        for j, (p_name, p) in enumerate(self.predicates_dict.items()):
            for s in range(self.len_s):
                eta = self.eta_hat_js[j, s]
                x = self.S[p_name][s]
                function -= eta * (1 - p(x))

        self.objective_function = cp.Minimize(function)

        return self.objective_function

    
    def construct_constraints(self):
        """
        制約不等式の作成
        """

        pointwise = self._construct_pointwise_constraints()
        logical = self._construct_logical_constraints()
        consistency = self._construct_consistency_constraints()

        constraints = pointwise + logical + consistency

        return constraints


    def main(self, c1=2.5, c2=2.5):
        """
        目的関数と制約の構成．
        """

        print('Loading data ...')
        s_time_1 = time.time()
        self.load_data()
        e_time_1 = time.time()
        print(f'Done in {e_time_1 - s_time_1} seconds! \n')
        
        print('Loading rules ...')
        s_time_2 = time.time()
        self.load_rules()
        e_time_2 = time.time()
        print(f'Done in {e_time_2 - s_time_2} seconds! \n')
        
        print('Constructing predicates ...')
        s_time_3 = time.time()
        self.create_predicates_with_cvxpy()
        e_time_3 = time.time()
        print(f'Done in {e_time_3 - s_time_3} seconds! \n')

        print('Getting coefficients of affine functions ...')
        s_time_4 = time.time()
        self.get_M_and_q()
        e_time_4 = time.time()
        print(f'Done in {e_time_4 - s_time_4} seconds! \n')

        print('Calculating predicates on all possible groundings ...')
        s_time_5 = time.time()
        self.evaluate_predicates_for_logical_constraints()
        e_time_5 = time.time()
        print(f'Done in {e_time_5 - s_time_5} seconds! \n')

        print('Constructing objective function ...')
        s_time_6 = time.time()
        obj_func = self.construct_objective_function(c1, c2)
        e_time_6 = time.time()
        print(f'Done in {e_time_6 - s_time_6} seconds! \n')
        







        print('Constructing constraints ...')
        s_time_5 = time.time()
        constraints = self.construct_constraints()
        e_time_5 = time.time()
        print(f'Done in {e_time_5 - s_time_5} seconds! \n')
        
        print('All done')

        return obj_func, constraints
    





import sympy as sp

from .operators import negation
from .operators import Semantisize_symbols
from .misc import is_symbol, process_neg

# symbols_1 = ['¬', '∧', '∨', '⊗', '⊕', '→']

symbols_tmp = Semantisize_symbols()
symbols_1_semanticized = symbols_tmp.symbols_1_semanticized
symbols_3_semanticized = symbols_tmp.symbols_3_semanticized
symbols = list(symbols_1_semanticized.keys()) + list(symbols_3_semanticized.keys())


class FOLConverter:

    def __init__(self, file_path):
        self.file_path = file_path
        self.KB_origin = self._construct_KB()
        self.KB = None
        self.KB_tmp = None

        self.predicates_dict_tmp = None
        
    def _construct_KB(self):
        """
        KB の .txt ファイルを読み込む
        """    
        KB = []

        with open(self.file_path, 'r') as file:
            for line in file:
                formula = line.split()
                KB.append(formula)

        return KB     
    
    def _identify_predicates(self, KB):
        """
        KB 内の述語を特定し，
        各述語の係数を取り出すために
        sympy.Symbol で表現する
        """
        predicates = []

        for formula in KB:
            for item in formula:
                if item not in symbols and item not in predicates:
                    predicates.append(item)
        
        predicates_dict = {predicate: sp.Symbol(predicate) for predicate in predicates}

        return predicates_dict

    def _check_implication(self, formula):
        """
        formula (リスト) について，
        含意記号 '→' の数を調べ，
        その数が 1 以下になっているかを確認する
        """
        
        # 実質，ここには 1 つのインデックスしか入らない
        implication_idxs = []

        for i, item in enumerate(formula):
            if item == '→':
                implication_idxs.append(i)
        
        implication_num = len(implication_idxs)
        
        if implication_num == 0:
            return False, None
        elif implication_num == 1:
            implication_idx = implication_idxs[0]
            return True, implication_idx
        else:
            print('this formula may be invalid')

    def _eliminate_implication(self, formula):
        """
        formula (リスト) 内に含意記号 '→' あれば変換し，消去する 
        """
        implication_flag, target_idx = self._check_implication(formula)

        if implication_flag:
            # 含意記号 '→' を境に formula (list) を 2 つに分ける
            x = formula[:target_idx]
            y = formula[target_idx + 1:]

            # x → y = ¬ x ⊕ y
            x_new = negation(x)
            y_new = y
            new_operator = ['⊕']

            new_formula = x_new + new_operator + y_new
        else:
            new_formula = formula

        return new_formula
    
    def _drop_x(self, formula):
        """
        formula (リスト) 内の '(x)' を削除する
        """
        new_formula = []
        for item in formula:
            new_item = item.replace("(x)", "")
            new_formula.append(new_item)

        return new_formula
       
    def _get_idx_list(self, formula):
        """
        formula (リスト) 内の '¬' のインデックスリストと
        '¬' 以外のインデックスリストを返す
        """
        neg_idxs = []
        not_neg_idxs = []

        for i, item in enumerate(formula):
            if item == '¬':
                neg_idxs.append(i)
            else:
                not_neg_idxs.append(i)
        
        return neg_idxs, not_neg_idxs

    def _split_idx_list(self, idx_list):
        """
        インデックスリストを連続する部分リストに分割する
        """
        result = []
        tmp = []

        for i in range(len(idx_list)):
            if not tmp or idx_list[i] == tmp[-1] + 1:
                tmp.append(idx_list[i])
            else:
                result.append(tmp)
                tmp = [idx_list[i]]

        if tmp:
            result.append(tmp)

        return result
    
    def _eliminate_multi_negations(self, formula):
        """
        formula (リスト) 内の連続する '¬' を削除する
        """
        neg_idxs, not_neg_idxs = self._get_idx_list(formula)
        neg_idxs_decomposed = self._split_idx_list(neg_idxs)

        neg_idxs_new = []
        for tmp in neg_idxs_decomposed:
            if len(tmp) % 2 == 0:
                pass
            else:
                neg_idxs_new.append(tmp[0])
        
        idxs_new = sorted(neg_idxs_new + not_neg_idxs)
        
        formula_new = []
        for idx in idxs_new:
            item = formula[idx]
            formula_new.append(item)

        return formula_new

    
    def main(self):
        """
        KB,
        KB_tmp,
        predicates_dict

        をそれぞれ計算する
        """
        self.KB = []
        for formula in self.KB_origin:
            new_formula = self._eliminate_multi_negations(formula)
            new_formula = self._eliminate_implication(new_formula)
            new_formula = self._drop_x(new_formula)
            self.KB.append(new_formula)

        self.predicates_dict_tmp = self._identify_predicates(self.KB)

        self.KB_tmp = []
        for formula in self.KB:

            tmp_formula = []
            for item in formula:
                if item in self.predicates_dict_tmp.keys():
                    tmp_formula.append(self.predicates_dict_tmp[item])
                else:
                    tmp_formula.append(item)
                
            process_neg(tmp_formula)

            phi_h = []
            new_formula_1 = [sp.Symbol('1')]
            new_formula_2 = []

            tmp_new_formula_2 = 0
            for item in tmp_formula:
                if not is_symbol(item):
                    tmp_new_formula_2 += item
            
            new_formula_2.append(tmp_new_formula_2)

            phi_h.append(new_formula_1)
            phi_h.append(new_formula_2)
        
            self.KB_tmp.append(phi_h)

