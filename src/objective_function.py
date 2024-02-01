from __future__ import annotations

import cvxpy as cp

from .misc import timer, is_symbol
from .operators import negation

# from .setup_problem import Setup
class Setup_:
    """
    型ヒント用（circular import の回避のため）
    """
    def __init__(self):
        pass


@timer
def linear_svm(obj: Setup_, 
               c1: float = 10, 
               c2: float = 10) -> cp.Expression:
    """
    目的関数を構成する．
    c1 は logical constraints を，
    c2 は consistency constraints を
    満足する度合いを表す．
    """

    len_j = obj.len_j
    len_l = obj.len_l
    len_h = obj.len_h
    w_j   = obj.w_j
    xi_jl = obj.xi_jl
    xi_h  = obj.xi_h

    print(f'obj coeff')
    print(f'c1: {c1}')
    print(f'c2: {c2}')

    function = 0

    for j in range(len_j):
        w = w_j[j, :-1]
        function += 1/2 * (cp.norm2(w) ** 2)

    for j in range(len_j):
        for l in range(len_l):
            xi = xi_jl[j, l]
            function += c1 * xi

    for h in range(len_h):
        xi = xi_h[h, 0]
        function += c2 * xi

    objective_function = cp.Minimize(function)

    return objective_function


@timer
def linear_svm_loss(obj: Setup_, 
                    c1: float = 10, 
                    c2: float = 10) -> cp.Expression:
    """
    目的関数を構成する．
    c1 は logical constraints を，
    c2 は consistency constraints を
    満足する度合いを表す．
    """
    len_j = obj.len_j
    len_l = obj.len_l
    w_j   = obj.w_j

    predicates_dict = obj.predicates_dict

    print(f'obj coeff')
    print(f'c1: {c1}')
    print(f'c2: {c2}')

    L = obj.L
    U = next(iter(obj.U.values())) # 仮

    function = 0

    for j in range(len_j):
        w = w_j[j, :-1]
        function += 1/2 * (cp.norm2(w) ** 2)

    for j, (p_name, p) in enumerate(predicates_dict.items()):
        for l in range(len_l):
            x = L[p_name][l, :-1]
            y = L[p_name][l, -1]

            tmp = cp.maximum(0, 1 - y * (2 * p(x) - 1))
            function += c1 * tmp

    for u in U:
        KB_tmp = obj._calc_KB_at_datum(obj.KB, u)
        for h, formula in enumerate(KB_tmp):

            formula_tmp = 0
            for item in formula:
                if not is_symbol(item):
                    formula_tmp += item

            tmp = cp.maximum(0, negation(formula_tmp))

            function += c2 * tmp

    objective_function = cp.Minimize(function)

    return objective_function


# logistic regression
@timer
def logistic_regression_loss(obj: Setup_, 
                             c1: float = 10, 
                             c2: float = 10) -> cp.Expression:
    """
    目的関数を構成する．
    c1 は logical constraints を，
    c2 は consistency constraints を
    満足する度合いを表す．
    """
    len_j = obj.len_j
    len_l = obj.len_l
    w_j   = obj.w_j

    predicates_dict = obj.predicates_dict

    print(f'obj coeff')
    print(f'c1: {c1}')
    print(f'c2: {c2}')

    L = obj.L
    U = next(iter(obj.U.values())) # 仮

    function = 0

    for j in range(len_j):
        w = w_j[j, :-1]
        function += 1/2 * (cp.norm2(w) ** 2)

    from .misc import log_loss
    for p_name, p in obj.predicates_dict.items():
        x = obj.L[p_name][:, :-1]
        y = obj.L[p_name][:, -1]
        y_pred = p(x)
        value = log_loss(y, y_pred)
        function += c1 * value 

    for u in U:
        KB_tmp = obj._calc_KB_at_datum(obj.KB, u)
        for h, formula in enumerate(KB_tmp):

            formula_tmp = 0
            for item in formula:
                if not is_symbol(item):
                    formula_tmp += item

            tmp = cp.maximum(0, negation(formula_tmp))

            function += c2 * tmp

    objective_function = cp.Minimize(function)

    return objective_function






