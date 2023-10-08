import os

import numpy as np
import pandas as pd
import cvxpy as cp




# データの次元が増えたら，重みとデータの内積を取るように書き換えて対応
# p の取る引数の数が同一でない問題設定もあるので注意
class Predicate:
    def __init__(self, w):
        self.w1 = w[0]
        self.w2 = w[1]
        self.b = w[2]

    def func(self, x):
        x1, x2 = x[0], x[1]
        return self.w1 * x1 + self.w2 * x2 + self.b
    



# cvxpy.Variable と str が混ざると，リストに対する組み込み関数での操作でエラーが出たため実装
def count_neg(formula_decomposed):
    neg_num = 0
    
    for item in formula_decomposed:
        if type(item) == str:
            if item == '¬':
                neg_num += 1

    return neg_num


def get_first_neg_index(formula_decomposed):
    target_index = None

    for i, item in enumerate(formula_decomposed):
        if type(item) == str:
            if item == '¬':
                target_index = i
                break
    
    return target_index



# 良い関数名が思い浮かばないので，仮実装とする
def count_(formula_decomposed):
    num_ = 0
    for item in formula_decomposed:
        if type(item) == str:
            if item in ['⊕', '→']:
                num_ += 1
    
    return num_


def neg_for_list(formula):
    # とりあえず演算子は weak な disj と conj しか許さない

    formula_tmp = []
    for item in formula:
        if item == '∧':
            formula_tmp.append('∧')
        elif item == '∨':
            formula_tmp.append('∨')
        else:
            formula_tmp.append(1 - item)
    
    formula = formula_tmp
    # return formula








########################## 仮実装    
def load_data():
    # load and convert data, describe problem settings, etc
    data_dir_path = '/home/onoue/ws/lukasiewicz/inputs/toy_data/'
    path_L1 = os.path.join(data_dir_path, 'L1.csv')
    path_L2 = os.path.join(data_dir_path, 'L2.csv')
    path_L3 = os.path.join(data_dir_path, 'L3.csv')
    path_U = os.path.join(data_dir_path, 'U.csv')

    df_L1 = pd.read_csv(path_L1, index_col=0)
    df_L2 = pd.read_csv(path_L2, index_col=0)
    df_L3 = pd.read_csv(path_L3, index_col=0)
    df_U = pd.read_csv(path_U, index_col=0)

    L1 = np.array(df_L1)
    L2 = np.array(df_L2)
    L3 = np.array(df_L3)

    L = np.stack([L1, L2, L3]) # data for pointwise constraint
    U = np.array(df_U) # data for logical constriant

    len_j = 3 # number of tasks (p の数)
    len_h = 2 # number of logical constraints considered (cardinality of KB)
    len_jl = 0 # number of pointwise constraints to be counted later


    len_l_list = [] # L_j の要素数のリスト
    len_s_list = [] # S_j の要素数のリスト
    S = [] # data for consistency constraints 

    for i in range(len_j):
        if len_h != 0:
            u = len(U)
            S_i = np.concatenate((L[i][:, :2], U), axis=0)
            S.append(S_i)
        else:
            u = 0
            S_i = L[i][:, :2]
            S.append(S_i)
        len_l_list.append(len(L[i]))
        len_jl += len(L[i])
        len_s_list.append(len(S_i))

    S = np.stack(S)

    return L, U, S



def boundary_equation_2d(x1, coeff):
    w1 = coeff[0]
    w2 = coeff[1]
    b = coeff[2]

    x = np.hstack([x1, np.ones_like(x1)])
    # w = np.array([-w1/w2, -b/w2]).reshape(-1,1)
    w = np.array([-w1/w2, -b/w2 + 0.5/w2]).reshape(-1,1)

    return x @ w





