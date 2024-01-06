import os
from typing import List, Dict, Tuple, Union
import sympy as sp
import cvxpy as cp
import pandas as pd
import numpy as np
from .operators import negation
from .misc import process_neg, is_symbol, timer, Predicate



# from .setup_problem import Setup
class Setup_:
    """
    型ヒント用（circular import の回避のため）
    """
    # def __init__(self):
    #     pass
    
    # FOLConverter の簡易動作確認用
    def __init__(self):
        self.len_j = 3
        self.len_u = 6
        

class FOLConverter:

    def __init__(self, obj: Setup_, file_path: str) -> None:
        self.obj = obj

        self.file_path = file_path
        self.KB_origin = self._construct_KB()
        self.KB = None

        self.KB_info = None

        self.M = None
        self.q = None

        self.predicates_dict_tmp = None
        
    def _construct_KB(self) -> List[List[str]]:
        """
        KB の .txt もしくは .tsv ファイルを読み込む
        """    
        KB_origin = []
        
        if "tsv" in self.file_path:
            KB_info = pd.read_table(self.file_path)
            rules = KB_info['formula']

            for line in rules:
                formula = line.split()
                KB_origin.append(formula)
        else:
            with open(self.file_path, 'r') as file:
                for line in file:
                    formula = line.split()
                    KB_origin.append(formula)

        return KB_origin

    def _identify_predicates(self, KB: List[List[str]]) -> Dict[str, sp.Symbol]:
        """
        KB 内の述語を特定し，
        各述語の係数を取り出すために
        sympy.Symbol で表現する
        """
        predicates = []

        for formula in KB:
            for item in formula:
                # if item not in ['¬', '∧', '∨', '⊗', '⊕', '→'] and item not in predicates:
                if not is_symbol(item) and item not in predicates:
                    predicates.append(item)
        
        predicates_dict = {predicate: sp.Symbol(predicate) for predicate in predicates}

        return predicates_dict

    # def _check_implication(self, formula: List[Union[str, sp.Symbol]]):
    def _check_implication(self, formula: List[str]) -> Tuple[bool, Union[None, int]]:
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

    def _eliminate_implication(self, formula: List[str]) -> List[str]:
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
       
    def _get_neg_idx_list(self, formula: List[str]) -> Tuple[List[str], List[str]]:
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

    def _split_neg_idx_list(self, idx_list) -> List[List[int]]:
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
    
    def _eliminate_multi_negations(self, formula: List[str]) -> List[str]:
        """
        formula (リスト) 内の連続する '¬' を削除する
        """
        neg_idxs, not_neg_idxs = self._get_neg_idx_list(formula)
        neg_idxs_decomposed = self._split_neg_idx_list(neg_idxs)

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
    
    def convert_KB_origin(self) -> None:
        self.KB = []
        for formula in self.KB_origin:
            new_formula = self._eliminate_multi_negations(formula)
            new_formula = self._eliminate_implication(new_formula)
            self.KB.append(new_formula)
    
    def calculate_M_and_q(self) -> None:
        self.predicates_dict_tmp = self._identify_predicates(self.KB)

        # sympy で predicate を構成した KB
        # （formula を式変形した後の predicate の係数を取り出すため）
        KB_tmp = []
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
        
            KB_tmp.append(phi_h)

        predicates = list(self.predicates_dict_tmp.values())

        self.M = []
        self.q = []

        for phi_h in KB_tmp:

            base_M_h = np.zeros((len(phi_h), self.obj.len_j))
            base_q_h = np.zeros((len(phi_h), 1))
            for i, formula in enumerate(phi_h):
                for j, predicate in enumerate(predicates):

                    # negation
                    val = sp.Symbol('1') - formula[0]
                    coefficient = val.coeff(predicate)
                    base_M_h[i, j] = coefficient
                
                # negation
                val = sp.Symbol('1') - formula[0]
                base_q_h[i] = val.coeff(sp.Symbol('1'))

            tmp_M_h = []
            for i in range(self.obj.len_j):
                column = base_M_h[:, i]
                zeros = np.zeros((len(phi_h), self.obj.len_u - 1))
                concatenated_column = np.concatenate((column[:, np.newaxis], zeros), axis=1)
                tmp_M_h.append(concatenated_column)

            tmp_M_h = [np.concatenate(tmp_M_h, axis=1)]
            
            shifted_M_h = tmp_M_h[0]

            for i in range(self.obj.len_u - 1):
                shifted_M_h = np.roll(shifted_M_h, 1, axis=1)
                tmp_M_h.append(shifted_M_h)
            
            M_h = np.concatenate(tmp_M_h, axis=0)
            self.M.append(M_h)

            
            tmp_q_h = [base_q_h for u in range(self.obj.len_u)]
            q_h = np.concatenate(tmp_q_h, axis=0)
            self.q.append(q_h)
        
        # self.M = np.array(tmp_M)
        # self.q = np.array(tmp_q)

    def main(self) -> Tuple[Union[None, pd.DataFrame], List[List[str]], List[List[str]], List[np.ndarray], List[np.ndarray], Dict[str, sp.Symbol]]:
        """
        KB,
        KB_tmp,
        predicates_dict

        をそれぞれ計算する
        """
        self.convert_KB_origin()
        self.calculate_M_and_q()

        return self.KB_info, self.KB_origin, self.KB, self.M, self.q, self.predicates_dict_tmp
        
        

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

    def __init__(self,
                 data_dir_path, 
                 file_names_dict, 
                 obj,
                 c1=2.5, c2=2.5):
        self.data_dir_path = data_dir_path
        self.file_names_dict = file_names_dict

        self.c1 = c1
        self.c2 = c2

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
        self.len_i = None

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
        self.obj = obj
        # self.objective_function = None


    @timer
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
            path = os.path.join(self.data_dir_path, file_name)
            self.L[file_name[2:-4]] = np.array(pd.read_csv(path, index_col=0))

        # self.U = {}
        # for file_name in self.file_names_dict['unsupervised']:
        #     path = os.path.join(self.data_dir_path, file_name)
        #     self.U[file_name[2:-4]] = np.array(pd.read_csv(path, index_col=0))

        # 教師なしデータ
        self.U = {}
        for file_name in self.file_names_dict['supervised']:
            path = os.path.join(self.data_dir_path, 'U.csv')
            self.U[file_name[2:-4]] = np.array(pd.read_csv(path, index_col=0))

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

        self.len_i = 2 * self.len_u

        S_tmp = next(iter(self.S.values()))
        self.len_s = len(S_tmp)

    @timer
    def load_rules(self):
        rule_path = os.path.join(self.data_dir_path, self.file_names_dict['rule'][0])
        fol_processor = FOLConverter(self, rule_path)
        self.KB_info, self.KB_origin, self.KB, self.M, self.q, self.predicates_dict = fol_processor.main()
        
        self.len_h = len(self.KB)

    def _define_cvxpy_variables(self):
        self.w_j = cp.Variable(shape=(self.len_j, self.dim_x_L))

        self.xi_jl = cp.Variable(shape=(self.len_j, self.len_l), nonneg=True)
        self.xi_h = cp.Variable(shape=(self.len_h, 1), nonneg=True)
        
        self.mu_jl = cp.Variable(shape=(self.len_j, self.len_l), nonneg=True)
        self.mu_h = cp.Variable(shape=(self.len_h, 1), nonneg=True)

        self.lambda_jl = cp.Variable(shape=(self.len_j, self.len_l), nonneg=True)
        self.lambda_hi = cp.Variable(shape=(self.len_h, self.len_i), nonneg=True)

        self.eta_js = cp.Variable(shape=(self.len_j, self.len_s), nonneg=True)
        self.eta_hat_js = cp.Variable(shape=(self.len_j, self.len_s), nonneg=True)

    @timer
    def formulate_predicates_with_cvxpy(self):
        """
        KB の中の全 predicate を取得して，辞書に格納．
        predicate function を作成して対応させる
        """
        predicates = self.predicates_dict.keys()
        self._define_cvxpy_variables()
        self.predicates_dict = {predicate: Predicate(self.w_j[j]) for j, predicate in enumerate(predicates)}
    
    @timer
    def construct_constraints(self):
        constraints = []

        for j in range(self.len_j):
            constraint_tmp = 0
            predicate_name = list(self.predicates_dict.keys())[j]
            for h in range(self.len_h):
                for i in range(self.len_i):
                    for u in range(self.len_u):
                        lmbda = self.lambda_hi[h ,i]
                        M = self.M[h][i, u]
                        constraint_tmp += lmbda * M

            for l in range(self.len_l):
                lmbda = self.lambda_jl[j, l]
                y = self.L[predicate_name][l, -1]
                constraint_tmp += -2 * lmbda * y

            for s in range(self.len_s):
                eta = self.eta_js[j, s]
                eta_hat = self.eta_hat_js[j, s]
                constraint_tmp += -1 * (eta - eta_hat)
            
            constraints += [
                constraint_tmp == 0
            ]
        
        for j in range(self.len_j):
            for l in range(self.len_l):
                lmbda = self.lambda_jl[j, l]
                constraints += [
                    lmbda >= 0,
                    lmbda <= self.c1
                ]
        
        for h in range(self.len_h):
            for i in range(self.len_i):
                lmbda = self.lambda_hi[h, i]
                constraints += [
                    lmbda >= 0,
                    lmbda <= self.c2
                ]

        for j in range(self.len_j):
            for s in range(self.len_s):
                eta = self.eta_js[j, s]
                eta_hat = self.eta_hat_js[j, s]
                constraints += [
                    eta >= 0,
                    eta_hat >= 0
                ]

        return constraints
    
    def main(self):
        self.load_data()
        self.load_rules()
        self.formulate_predicates_with_cvxpy()
        objective_function = self.obj(self).construct()
        constraints = self.construct_constraints()
        return objective_function, constraints
        



class ObjectiveFunction:
    def __init__(self, obj: Setup_, kernel_function: object = None) -> None:
        self.L = obj.L
        self.U = obj.U
        self.S = obj.S

        self.len_j = obj.len_j
        self.len_l = obj.len_l
        self.len_u = obj.len_u
        self.len_s = obj.len_s
        self.len_h = obj.len_h 
        self.len_i = obj.len_i

        self.lambda_jl  = obj.lambda_jl
        self.lambda_hi  = obj.lambda_hi
        self.eta_js     = obj.eta_js
        self.eta_hat_js = obj.eta_hat_js

        self.M = obj.M 
        self.q = obj.q

        self.predicate_names = list(obj.predicates_dict.keys())

        if kernel_function == None:
            self.k = self.linear_kernel
        else:
            self.k = kernel_function

    def linear_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.dot(x1, x2)
    
    def _mapping_variables(self) -> Tuple[dict, List[cp.Variable]]:
        mapping_x_i = {}
        x = []

        mapping_x_i["lambda_jl"] = {}
        for j in range(self.len_j):
            for l in range(self.len_l):
                mapping_x_i['lambda_jl'][(j, l)] = len(x)
                x.append(self.lambda_jl[j, l])

        mapping_x_i["lambda_hi"] = {}
        for h in range(self.len_h):
            for i in range(self.len_i):
                mapping_x_i["lambda_hi"][(h, i)] = len(x)
                x.append(self.lambda_hi[h, i])

        mapping_x_i['delta_eta_js'] = {}
        for j in range(self.len_j):
            for s in range(self.len_s):
                mapping_x_i["delta_eta_js"][(j, s)] = len(x)
                x.append(self.eta_js[j, s] - self.eta_hat_js[j, s])
            
        return mapping_x_i, x 
    
    @timer
    def _construct_P(self) -> Tuple[cp.Variable, np.ndarray]: 
        mapping_x_i, x = self._mapping_variables()
        P = np.zeros((len(x), len(x)))

        for j in range(self.len_j):
            key = self.predicate_names[j]
            L = self.L[key]
            U = self.U[key]
            S = self.S[key]

            for l_row in range(self.len_l):
                row = mapping_x_i['lambda_jl'][(j, l_row)]
                x_row = L[l_row, :-1]
                y_row = L[l_row, -1]
                for l_col in range(self.len_l):
                    col = mapping_x_i['lambda_jl'][(j, l_col)]
                    x_col = L[l_col, :-1]
                    y_col = L[l_col, -1]
                    P[row, col] += 4 * y_row * y_col * self.k(x_row, x_col)
            
            for h_row in range(self.len_h):
                for h_col in range(self.len_h):
                    for i_row in range(self.len_i):
                        for i_col in range(self.len_i):
                            for u_row in range(self.len_u):
                                for u_col in range(self.len_u):
                                    M_row = self.M[h_row][i_row, u_row]
                                    M_col = self.M[h_col][i_col, u_col]
                                    if M_row != 0 and M_col != 0:
                                        row = mapping_x_i['lambda_hi'][(h_row, i_row)]
                                        col = mapping_x_i['lambda_hi'][(h_col, i_col)]
                                        x_row = U[u_row]
                                        x_col = U[u_col]
                                        P[row, col] += M_row * M_col * self.k(x_row, x_col)

            for s_row in range(self.len_s):
                row = mapping_x_i["delta_eta_js"][j, s_row]
                x_row = S[s_row]
                for s_col in range(self.len_s):
                    col = mapping_x_i["delta_eta_js"][j, s_col]
                    x_col = S[s_col]
                    P[row, col] += self.k(x_row, x_col)
                
            for l in range(self.len_l):
                row = mapping_x_i["lambda_jl"][(j, l)]
                x_l = L[l, :-1]
                y_l = L[l, -1]
                for h in range(self.len_h):
                    for i in range(self.len_i):
                        col = mapping_x_i["lambda_hi"][(h, i)]
                        for u in range(self.len_u):
                            x_u = U[u]
                            M = self.M[h][i, u]
                            if M != 0:
                                P[row, col] += -4 * y_l * M * self.k(x_l, x_u)
            
            for l in range(self.len_l):
                row = mapping_x_i["lambda_jl"][(j, l)]
                x_l = L[l, :-1]
                y_l = L[l, -1]
                for s in range(self.len_s):
                    col = mapping_x_i["delta_eta_js"][(j, s)]
                    x_s = S[s]
                    P[row, col] += 4 * y_l * self.k(x_l, x_s)
                
            for h in range(self.len_h):
                for i in range(self.len_i):
                    row = mapping_x_i["lambda_hi"][(h, i)]
                    for s in range(self.len_s):
                        col = mapping_x_i["delta_eta_js"][(j, s)]
                        x_s = S[s]
                        for u in range(self.len_u):
                            x_u = U[u]
                            M = self.M[h][i, u]
                            if M != 0:
                                P[row, col] += -2 * M * self.k(x_u, x_s)
        
        P = (-1/2)*(P+P.T)/2
        return cp.vstack(x), P
    
    # def _construct_q_(self, mapping_x_i: Dict[str, dict]) -> np.ndarray:
    #     q_ = np.zeros((len(x), 1))
    #     col = 0

    #     for j in range(self.len_j):
    #         for l in range(self.len_l):
    #             row = mapping_x_i["lambda_jl"][(j, l)]
    #             q_[row, col] += 1

    #             print(row)
        
    #     for h in range(self.len_h):
    #         for i in range(self.len_i):
    #             row = mapping_x_i["lambda_hi"][(h, i)]
    #             for u in range(self.len_u):
    #                 M = self.M[h][i, u]
    #                 q = self.q[h][i, 0]
    #                 q_[row, col] += (1/2) * M + q

    #             print(row)
        
    #     for j in range(self.len_j):
    #         for s in range(self.len_s):
                 

    def construct(self) -> cp.Expression:
        x, P = self._construct_P()
        objective_function = cp.quad_form(x, P)

        for j in range(self.len_j):
            for l in range(self.len_l):
                objective_function += self.lambda_jl[j, l]
        
        for h in range(self.len_h):
            for i in range(self.len_i):
                for u in range(self.len_u):
                    objective_function += self.lambda_hi[h, i] * (1/2 * self.M[h][i, u] + self.q[h][i, 0])
            
        for j in range(self.len_j):
            for s in range(self.len_s):
                objective_function += (-1/2) * (self.eta_js[j, s] + self.eta_hat_js[j, s])

        objective_function = cp.Maximize(objective_function)

        return objective_function
    

class Predicate_dual:
    def __init__(self, obj: Setup_, name: str, kernel_function: object = None) -> None:
        self.c1 = obj.c1
        self.c2 = obj.c2

        self.L = obj.L[name]
        self.U = obj.U[name]
        self.S = obj.S[name]

        self.len_l = obj.len_l
        self.len_u = obj.len_u
        self.len_s = obj.len_s
        self.len_h = obj.len_h 
        self.len_i = obj.len_i

        self.p_idx = list(obj.predicates_dict.keys()).index(name)

        self.lambda_jl  = obj.lambda_jl[self.p_idx, :].value
        self.lambda_hi  = obj.lambda_hi.value
        self.eta_js     = obj.eta_js[self.p_idx, :].value
        self.eta_hat_js = obj.eta_hat_js[self.p_idx, :].value

        start_col = self.p_idx * self.len_u
        end_col = start_col + self.len_u
        # self.M = [M_h[:, start_col:end_col] for M_h in obj.M]
        self.M = obj.M

        if kernel_function == None:
            self.k = self.linear_kernel
        else:
            self.k = kernel_function

        ##########################################################
        ##########################################################
        ##########################################################
        """ 
        なぜか b += 0.5 するとうまく識別できている．（p の予測値に対して 0.5 が境界）
        入力データで y in {-1, 1} であるからか
        """
        self.b = self._b() + 0.5
        # self.b = self._b()

        self.w_linear_kernel = self._w_linear_kernel() 

        self.coeff = np.append(self.w_linear_kernel, self.b)


    def linear_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.dot(x1, x2)
    
    def w_dot_phi(self, x_pred: np.ndarray) -> float:
        k = self.k
        value = 0

        for l in range(self.len_l):
            x = self.L[l, :-1]
            y = self.L[l, -1]
            lmbda = self.lambda_jl[l]
            value += 2 * lmbda * y * k(x, x_pred)

        for h in range(self.len_h):
            for i in range(self.len_i):
                lmbda = self.lambda_hi[h, i]
                for u in range(self.len_u):
                    x = self.U[u]
                    M = self.M[h][i, u]
                    value += - lmbda * M * k(x, x_pred)
        
        for s in range(self.len_s):
            x = self.S[s]
            eta = self.eta_js[s]
            eta_hat = self.eta_hat_js[s]
            value += (eta - eta_hat) * k(x, x_pred)

        return value
    
    def _w_linear_kernel(self) -> np.ndarray:
        input_dim = len(self.L[0, :-1])
        w_linear_kernel = np.zeros(input_dim)

        for l in range(self.len_l):
            x = self.L[l, :-1]
            y = self.L[l, -1]
            lmbda = self.lambda_jl[l]
            w_linear_kernel += 2 * lmbda * y * x

        for h in range(self.len_h):
            for i in range(self.len_i):
                lmbda = self.lambda_hi[h, i]
                for u in range(self.len_u):
                    x = self.U[u]
                    M = self.M[h][i, u]
                    w_linear_kernel += - lmbda * M * x
        
        for s in range(self.len_s):
            x = self.S[s]
            eta = self.eta_js[s]
            eta_hat = self.eta_hat_js[s]
            w_linear_kernel += (eta - eta_hat) * x

        return w_linear_kernel
    
    def _b(self) -> float:
        value_tmp = 0
        count = 0

        for l in range(self.len_l):
            lmbda = self.lambda_jl[l]

            if lmbda >= 0 and lmbda <= self.c1:
                x = self.L[l, :-1]
                y = self.L[l, -1] 

                value_tmp += y - self.w_dot_phi(x)
                count += 1

        value = value_tmp / count

        return value

    def __call__(self, x_pred: np.ndarray) -> float:
        
        value = self.w_dot_phi(x_pred) + self.b

        return value