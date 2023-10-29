import os

import numpy as np
import pandas as pd
import cvxpy as cp

import matplotlib.pyplot as plt

from .operators import negation
from .operators import Semantisize_symbols

symbols_tmp = Semantisize_symbols()
symbols_1_semanticized = symbols_tmp.symbols_1_semanticized
symbols_3_semanticized = symbols_tmp.symbols_3_semanticized
symbols = list(symbols_1_semanticized.keys()) + list(symbols_3_semanticized.keys())


class Predicate:
    """
    述語．formula の構成要素の 1 つ．
    p の取る引数の数が同一でない問題設定もあるようなので，
    そのときは修正が必要
    """
    def __init__(self, w):
        self.w = w

    def __call__(self, x):
        w = self.w[:-1]
        b = self.w[-1]
        return w @ x + b
    

def _count_neg(formula_decomposed):
    """
    process_neg 関数の中で使用．
    cvxpy.Variable と str が混ざると，
    リストに対する組み込み関数での操作でエラーが出たため実装
    list.count(x)

    formula (list) 内の '¬' の数を数える
    """
    neg_num = 0
    
    for item in formula_decomposed:
        if type(item) == str:
            if item == '¬':
                neg_num += 1

    return neg_num


def _get_first_neg_index(formula_decomposed):
    """
    process_neg 関数の中で使用．
    cvxpy.Variable と str が混ざると，
    リストに対する組み込み関数での操作でエラーが出たため実装
    list.index(x)

    formula (list) 内の初めの '¬' のインデックスを取得
    """
    target_index = None

    for i, item in enumerate(formula_decomposed):
        if type(item) == str:
            if item == '¬':
                target_index = i
                break
    
    return target_index


def process_neg(formula):
    """
    formula（list）に含まれている
    否定記号 '¬' を変換し，消去する 
    """
    neg_num = _count_neg(formula)

    while neg_num > 0:
        target_index = _get_first_neg_index(formula)

        # 演算に使用する値を取得
        x = formula[target_index + 1]

        # 演算の実行
        operation = symbols_1_semanticized['¬']
        result = operation(x)

        # 演算結果で置き換え，演算子（¬）の削除
        formula[target_index + 1] = result
        formula.pop(target_index)

        neg_num -= 1

    # return formula


def count_specific_operator(formula_decomposed, operator):
    """
    formula（list）について，
    特定の演算記号の数を数える
    """
    neg_num = 0
    
    for item in formula_decomposed:
        if type(item) == str:
            if item == operator:
                neg_num += 1

    return neg_num


def get_first_specific_oprator_index(formula_decomposed, operator):
    """
    formula (list) について，
    特定の演算記号のインデックスのうち，
    一番小さいものを取得
    """
    target_index = None

    for i, item in enumerate(formula_decomposed):
        if type(item) == str:
            if item == operator:
                target_index = i
                break
    
    return target_index

def is_symbol(item):
    """
    リストとして保持されている formula の要素が演算記号であるかを判定
    """
    flag = False

    if type(item) != str:
        return flag
    else:
        for symbol in symbols:
            if item == symbol:
                flag = True
        return flag


def boundary_equation_2d(x1, coeff):
    """
    境界条件の方程式
    入力データの次元が 2 のときのみ使用可能
    """
    w1 = coeff[0]
    w2 = coeff[1]
    b = coeff[2]

    x = np.hstack([x1, np.ones_like(x1)])
    w = np.array([-w1/w2, -b/w2 + 0.5/w2]).reshape(-1,1)

    return x @ w
    

def visualize_result(problem_instance, colors=['red', 'blue', 'green', 'yellow', 'black']):
    """
    入力データの次元が 2 のときのみ使用可能
    """
    L = problem_instance.L
    w_j = problem_instance.w_j.value
    len_j = problem_instance.len_j
    len_l = problem_instance.len_l

    test_x = np.linspace(0.05, 0.95, 100).reshape(-1, 1)
    test_ys = []
    for w in w_j:
        test_ys.append(boundary_equation_2d(test_x, w))

    plt.figure(figsize=(6,4))
    
    for j in range(len_j):
        for l in range(len_l):
            if L[j][l, 2] == 1:
                plt.scatter(L[j][l,0], L[j][l,1], c=colors[j], marker='o', label='1')
            else:
                plt.scatter(L[j][l,0], L[j][l,1], facecolors='none', edgecolors=colors[j], marker='o', label='-1')

    for j, test_y in enumerate(test_ys):
        plt.plot(test_x, test_y, label=f'p_{j+1}')
    
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.show()


