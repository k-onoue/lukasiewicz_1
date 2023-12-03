import os
import time

import cvxpy as cp
import numpy as np
import pandas as pd

from .operators import negation
from .operators import Semantisize_symbols

from .misc import process_neg, Predicate, is_symbol

# from .process_fol import FOLConverter


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

    def __init__(self, data_dir_path, file_names_dict, custom_obj_func_constructor):
        self.data_dir_path = data_dir_path
        self.file_names_dict = file_names_dict

        # データ
        self.L = None
        self.U = None
        self.S = None 

        # 仮
        self.KB_origin = None
        self.KB = None

        self.KB_info = None

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

        # custom objective function constructor 
        self.construct_objective_function = custom_obj_func_constructor


    def load_data(self):
        """
        .csv ファイルからデータを読み込んで，
        辞書を用いて，predicate 名でラベル付けをした
        ndarray として格納する

        {
        'p1(x)': np.array(),
        'p2(x)': np.array(),
        ...
        'pm(x)': np.array()
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

        S_tmp = next(iter(self.S.values()))
        self.len_s = len(S_tmp)


    def load_rules(self):
        """
        .txt ファイルとして保存されている Knowledge Base (KB) を読み込み，
        リストとして保持する
        """

        rules_path = os.path.join(self.data_dir_path, self.file_names_dict['rule'][0] + '.txt')
        fol_processor = FOLConverter(rules_path)
        self.KB_origin = fol_processor.KB
        self.KB = fol_processor.main()

        self.len_h = len(self.KB)

        self.KB_info = fol_processor.KB_info



    def _define_cvxpy_variables(self):
        # self.w_j = cp.Variable(shape=(self.len_j, 3))
        self.w_j = cp.Variable(shape=(self.len_j, self.dim_x_L))
        self.xi_jl = cp.Variable(shape=(self.len_j, self.len_l), nonneg=True)
        self.xi_h = cp.Variable(shape=(self.len_h, 1), nonneg=True)
    

    def identify_predicates(self):
        """
        KB の中の全 predicate を取得して，辞書に格納．
        predicate function を作成して対応させる
        """
        predicates = []

        KB = self.KB_origin
        for formula in KB:
            for item in formula:
                if item not in symbols and item not in predicates:
                    predicates.append(item)

        self.len_j = len(predicates)
        self._define_cvxpy_variables()

        self.predicates_dict = {predicate: Predicate(self.w_j[j]) for j, predicate in enumerate(predicates)}
        

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

    
    # def construct_objective_function(self, c1, c2):
    #     """
    #     目的関数を構成する．
    #     c1 は logical constraints を，
    #     c2 は consistency constraints を
    #     満足する度合いを表す．
    #     """

    #     function = 0

    #     for j in range(self.len_j):
    #         w = self.w_j[j]
    #         function += 1/2 * (cp.norm2(w) ** 2)

    #     for j in range(self.len_j):
    #         for l in range(self.len_l):
    #             xi = self.xi_jl[j, l]
    #             function += c1 * xi

    #     for h in range(self.len_h):
    #         xi = self.xi_h[h, 0]
    #         function += c2 * xi

    #     self.objective_function = cp.Minimize(function)

    #     return self.objective_function

    
    # def _construct_pointwise_constraints(self):
    #     """
    #     pointwise constraints を構成する
    #     """

    #     constraints_tmp = []

    #     for j, (p_name, p) in enumerate(self.predicates_dict.items()):
    #         for l in range(self.len_l):
    #             x = self.L[p_name][l, :-1]
    #             y = self.L[p_name][l, -1]

    #             xi = self.xi_jl[j, l]

    #             constraints_tmp += [
    #                 y * (2 * p(x) - 1) >= 1 - 2 * xi
    #             ]
        
    #     return constraints_tmp
    

    # def _construct_logical_constraints(self):
    #     """
    #     logical constraints を構成する
    #     """

    #     constraints_tmp = []

    #     # 仮
    #     U = next(iter(self.U.values()))

    #     for u in U:
    #         KB_tmp = self._calc_KB_at_datum(self.KB, u)           

    #         for h, formula in enumerate(KB_tmp):
          
    #             xi = self.xi_h[h]

    #             formula_tmp = 0
    #             for item in formula:
    #                 if not is_symbol(item):
    #                     formula_tmp += item

    #             constraints_tmp += [
    #                 0 <= xi,
    #                 negation(formula_tmp) <= xi,
    #             ]

    #     return constraints_tmp
    
    # def _construct_consistency_constraints(self):
    #     """
    #     consistency constraints を構成する
    #     """

    #     constraints_tmp = []
    #     for (p_name, p) in self.predicates_dict.items():
    #         for s in range(self.len_s):
    #             x = self.S[p_name][s]

    #             constraints_tmp += [
    #                 p(x) >= 0,
    #                 p(x) <= 1
    #             ]
        
    #     return constraints_tmp

    
    # def construct_constraints(self):
    #     """
    #     制約不等式の作成
    #     """

    #     pointwise = self._construct_pointwise_constraints()
    #     logical = self._construct_logical_constraints()
    #     consistency = self._construct_consistency_constraints()

    #     constraints = pointwise + logical + consistency

    #     return constraints


    def main(self, **kwargs):
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
        
        print('Identifying predicates ...')
        s_time_3 = time.time()
        self.identify_predicates()
        e_time_3 = time.time()
        print(f'Done in {e_time_3 - s_time_3} seconds! \n')
        
        print('Constructing objective function ...')
        s_time_4 = time.time()
        obj_func = self.construct_objective_function(self, **kwargs)
        e_time_4 = time.time()
        print(f'Done in {e_time_4 - s_time_4} seconds! \n')
        
        print('Constructing constraints ...')
        s_time_5 = time.time()
        constraints = self.construct_constraints(self)
        e_time_5 = time.time()
        print(f'Done in {e_time_5 - s_time_5} seconds! \n')
        
        print('All done')

        return obj_func, constraints
    


class FOLConverter:
    """
    Knowledge base (KB) を .txt ファイルで受け取り，
    その中の述語論理で記述された rule を
    '¬' と '⊕' だけの形に変換し，
    リストとして返す
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.KB = None
        self.new_KB = None

        self.KB_info = None

        self._construct_KB()
        
    def _construct_KB(self):
        """
        KB の .txt ファイルを読み込む
        """    
        self.KB_info = pd.read_table(self.file_path)
        rules = self.KB_info['formula']

        self.KB = []
        for line in rules:
            formula = line.split()
            self.KB.append(formula)


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
    

    def main(self):
        """
        新しい KB を返す
        """
        new_KB = []

        for formula in self.KB:
            # new_formula = self._eliminate_multi_negations(formula)
            new_formula = self._eliminate_implication(formula)
            new_KB.append(new_formula)

        self.new_KB = new_KB
        
        return new_KB
    

    def _get_idx_list(self, formula):
        neg_idxs = []
        not_neg_idxs = []

        for i, item in enumerate(formula):
            if item == '¬':
                neg_idxs.append(i)
            else:
                not_neg_idxs.append(i)
        
        return neg_idxs, not_neg_idxs


    def _split_idx_list(self, idx_list):
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
