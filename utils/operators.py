import cvxpy as cp


# # ¬ 否定
# def negation(x):
#     return 1 - x

# ¬ 否定
def negation(x):
    if type(x) != list:
        return 1 - x
    else:
        formula = []
        for i, item in enumerate(x):
            # if item == '∧':
            #     formula.append('∨')
            # elif item == '∨':
            #     formula.append('∧')
            # elif item == '⊕':
            #     formula.append('⊗')
            # elif item == '⊗':
            #     formula.append('⊕')
            # elif item == '¬':
            #     pass
            # elif item == '→':
            #     print("This may cause an error, please eliminate '→' first.")
            # else:
                # if x[i-1] == '¬':
                #     formula.append(item)
                # else:
            formula.append('¬')
            formula.append(item)

        # print('something wrong')
        # return None
        return formula

# ∧ 論理積 and
def weak_conjunction(x, y):
    return cp.minimum(x, y)

# ∨ 論理和 or 
def weak_disjunction(x, y):
    return cp.maximum(x, y)

# ⊗ 排他的論理積 （ではない）
def strong_conjunction(x, y):
    return cp.maximum(0, x + y - 1)

# ⊕ 排他的論理和 （ではない）
def strong_disjunction(x, y):
    return cp.minimum(1, x + y)

# → 含意
def implication(x, y):
    return strong_disjunction(negation(x), y)

# 通常の和
def plus(x, y):
    return x + y

# 通常の差
def minus(x, y):
    return x - y



class Semantisize_symbols:
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



    
        




