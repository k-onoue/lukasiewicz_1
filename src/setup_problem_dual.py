import os
from typing import List, Dict, Tuple, Union
import sympy as sp
import cvxpy as cp
import pandas as pd
import numpy as np
from .operators import negation
from .misc import process_neg, is_symbol, timer, Predicate
from .predicate import Predicate_dual
from .preprocess_fol import FOLConverter
from .objective_function import ObjectiveFunction

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

            start_col = j * self.len_u
            end_col = start_col + self.len_u
            M_j = [M_h[:, start_col:end_col] for M_h in self.M]

            constraint_tmp = 0
            predicate_name = list(self.predicates_dict.keys())[j]
            for h in range(self.len_h):
                for i in range(self.len_i):
                    for u in range(self.len_u):
                        lmbda = self.lambda_hi[h ,i]
                        # M = self.M[h][i, u]
                        M = M_j[h][i, u]
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

        # constraint_tmp = 0
        # for j in range(self.len_j):

        #     start_col = j * self.len_u
        #     end_col = start_col + self.len_u
        #     M_j = [M_h[:, start_col:end_col] for M_h in self.M]

        #     predicate_name = list(self.predicates_dict.keys())[j]
        #     for h in range(self.len_h):
        #         for i in range(self.len_i):
        #             for u in range(self.len_u):
        #                 lmbda = self.lambda_hi[h ,i]
        #                 # M = self.M[h][i, u]
        #                 M = M_j[h][i, u]
        #                 constraint_tmp += lmbda * M

        #     for l in range(self.len_l):
        #         lmbda = self.lambda_jl[j, l]
        #         y = self.L[predicate_name][l, -1]
        #         constraint_tmp += -2 * lmbda * y

        #     for s in range(self.len_s):
        #         eta = self.eta_js[j, s]
        #         eta_hat = self.eta_hat_js[j, s]
        #         constraint_tmp += -1 * (eta - eta_hat)
        
        # constraints += [
        #     constraint_tmp == 0
        # ]
        
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