import cvxpy as cp

# ¬ 否定
def negation(x):
    return 1 - x

# ∧ 論理積 and
def weak_conjunction(x, y):
    return cp.minimum(x, y)

# ∨ 論理和 or 
def weak_disjunction(x, y):
    return cp.maximum(x, y)

# ⊗ 排他的論理積
def strong_conjunction(x, y):
    return cp.maximum(0, x + y - 1)

# ⊕ 排他的論理和
def strong_disjunction(x, y):
    return cp.minimum(1, x + y)

# → 含意
def implication(x, y):
    return strong_disjunction(negation(x), y)