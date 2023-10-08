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
        for item in x:
            if item == '∧':
                formula.append('∨')
            elif item == '∨':
                formula.append('∧')
            else:
                neg_item = 1 - item
                formula.append(neg_item)

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






























def formula_transformer(formula_decomposed):
    neg_num = count_neg(formula_decomposed)

    while neg_num > 0:
        target_index = get_first_neg_index(formula_decomposed)

        # 演算に使用する値を取得
        x = formula_decomposed[target_index + 1]

        # 演算の実行
        operation = symbols_1_semanticized['¬']
        result = operation(x)

        # 演算結果で置き換え，演算子（¬）の削除
        formula_decomposed[target_index + 1] = result
        formula_decomposed.pop(target_index)

        neg_num -= 1

    while len(formula_decomposed) > 1:
        target_index = None

        # 分割された formula （リスト）から最初の演算子のインデックスを取得
        for index, token in enumerate(formula_decomposed):
            if token in symbols_1_semanticized.keys():
                target_index = index
                break

        # 対応する演算子の関数を取得
        symbol = formula_decomposed[target_index]
        operation = symbols_1_semanticized[symbol]

        # 演算に使用する値のペアを取得
        x = formula_decomposed[target_index - 1]
        y = formula_decomposed[target_index + 1]

        # 実際に演算を行う
        result = operation(x, y)

        # 演算結果で置き換え，演算済み要素の削除

        formula_decomposed[target_index] = result
        indices_to_remove = [target_index - 1, target_index + 1]
        filtered = []

        for i, item in enumerate(formula_decomposed):
            if i not in indices_to_remove:
                filtered.append(item)

        formula_decomposed = filtered

    return formula_decomposed[0]