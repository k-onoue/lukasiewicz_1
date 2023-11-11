import cvxpy as cp


"""
二重否定のところの処理が間違っているが，
奇跡的に， Toy Problem と winston で上手く行っている
"""
def negation(x):
    """
    ¬ 否定:
    この関数は以下の 2 つの場合に使用する．
    1. predicate(x) の否定を取る
    2. formula (list) の否定を取る 
    """
    if type(x) != list:
        return 1 - x
    else:
        formula = []
        flag_neg = False
        for i, item in enumerate(x):
            if item == '∧':
                formula.append('∨')
            elif item == '∨':
                formula.append('∧')
            elif item == '⊕':
                formula.append('⊗')
            elif item == '⊗':
                formula.append('⊕')
            elif item == '¬':
                flag_neg = True
                pass
            elif item == '→':
                print("This may cause an error, please eliminate '→' first.")
            # else:
            #     # if x[i-1] == '¬':
            #     #     formula.append(item)
            #     # else:
            #     #     formula.append('¬')
            #     #     formula.append(item)
            #     formula.append('¬')
            #     formula.append(item)

            else:
                if flag_neg == False:
                    formula.append('¬')
                    formula.append(item)
                else:
                    formula.append(item)
                    flag_neg = False

        return formula

def weak_conjunction(x, y):
    """
    ∧ or:
    x ∧ y = min{x, y} 
    """
    return cp.minimum(x, y)

def weak_disjunction(x, y):
    """
    ∨ and:
    x ∨ y = max{x, y}
    """
    return cp.maximum(x, y)

def strong_conjunction(x, y):
    """
    ⊗ t-norm:
    x ⊗ y = max{0, x + y - 1}
    """
    return cp.maximum(0, x + y - 1)

def strong_disjunction(x, y):
    """
    ⊕ o plus:
    x ⊕ y = min{1, x + y}
    """
    return cp.minimum(1, x + y)

def implication(x, y):
    """
    → 含意:
    x → y = min{1, 1 - x + y} 
        = min{1, (1 - x) + y} 
        = (1 - x) ⊕ y
        = ¬ x ⊕ y
    """
    return strong_disjunction(negation(x), y)

def plus(x, y):
    """
    + 通常の和
    """
    return x + y

def minus(x, y):
    """
    - 通常の差
    """
    return x - y


class Semantisize_symbols:
    """
    演算記号にその演算規則を意味付けする.
    ただし，symbols_2 および symbols_3 
    は現在使用されていない
    """

    def __init__(self):
        self.symbols_1 = ['¬', '∧', '∨', '⊗', '⊕', '→']
        self.symbols_2 = ['∀', '∃']
        self.symbols_3 = ['+', '-']

        self.operations_1 = [
            negation,
            weak_conjunction,
            weak_disjunction,
            strong_conjunction,
            strong_disjunction,
            implication
        ]

        self. operations_3 = [
            plus, 
            minus
        ]

        self.symbols_1_semanticized = {s: o for s, o in zip(self.symbols_1, self.operations_1)}
        self.symbols_3_semanticized = {s: o for s, o in zip(self.symbols_3, self.operations_3)}

