from __future__ import annotations

from typing import Dict, List, Union

from .operators import negation
from .misc import is_symbol, timer

import cvxpy as cp


# from .setup_problem import Setup
class Setup_:
    """
    型ヒント用（circular import の回避のため）
    """
    def __init__(self):
        pass


class ConstraintsConstructor:
    """
    制約を生成する．
    インスタンス化の際に flags に True または False を指定することによって,
    含める制約のタイプを選択できる．
    何も指定しない場合は pointwise, logical, consistency の全てを自動的に含む．

    例．pointwise と logical を含めて，consistency を含めない．
    constraint_flag_dict = {
        "pointwise"  : True,
        "logical"    : True,
        "consistency": False
    }
    """
    def __init__(self, 
                 obj: Setup_, 
                 flags: Dict[str, bool] = None) -> None:
        self.obj = obj

        if not flags:
            self.pointwise_flag   = True
            self.logical_flag     = True
            self.consistency_flag = True
        else:
            self.pointwise_flag   = flags['pointwise']
            self.logical_flag     = flags['logical']
            self.consistency_flag = flags['consistency']

    def _construct_pointwise_constraints(self) -> List[cp.Expression]:
        """
        pointwise constraints を構成する
        """

        constraints_tmp = []

        for j, (p_name, p) in enumerate(self.obj.predicates_dict.items()):
            for l in range(self.obj.len_l):
                x = self.obj.L[p_name][l, :-1]
                y = self.obj.L[p_name][l, -1]

                xi = self.obj.xi_jl[j, l]

                constraints_tmp += [
                    y * (2 * p(x) - 1) >= 1 - 2 * xi
                ]

        print("pointwise constraints")

        return constraints_tmp
    

    def _construct_logical_constraints(self) -> List[cp.Expression]:
        """
        logical constraints を構成する
        """

        constraints_tmp = []

        # 仮
        U = next(iter(self.obj.U.values()))

        for u in U:
            KB_tmp = self.obj._calc_KB_at_datum(self.obj.KB, u)           

            for h, formula in enumerate(KB_tmp):
          
                xi = self.obj.xi_h[h]

                formula_tmp = 0
                for item in formula:
                    if not is_symbol(item):
                        formula_tmp += item

                constraints_tmp += [
                    0 <= xi,
                    negation(formula_tmp) <= xi,
                ]

        print("logical constraints")

        return constraints_tmp
    
    def _construct_consistency_constraints(self) -> List[cp.Expression]:
        """
        consistency constraints を構成する
        """

        constraints_tmp = []
        for (p_name, p) in self.obj.predicates_dict.items():
            for s in range(self.obj.len_s):
                x = self.obj.S[p_name][s]

                constraints_tmp += [
                    p(x) <= 1,
                    p(x) >= -1,
                ]

        print("consistency constraints")
        
        return constraints_tmp

    
    # def construct_constraints(self):
    @timer
    def __call__(self) -> Union[List[cp.Expression], None]:
        """
        制約不等式の作成
        """

        if self.pointwise_flag:
            pointwise = self._construct_pointwise_constraints()
        else:
            pointwise = []

        if self.logical_flag:
            logical = self._construct_logical_constraints()
        else:
            logical = []

        if self.consistency_flag:
            consistency = self._construct_consistency_constraints()
        else:
            consistency = []

        constraints = pointwise + logical + consistency
        if not constraints:
            constraints = None

        return constraints

