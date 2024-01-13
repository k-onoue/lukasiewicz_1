# memo

- preprocess_fol.py の修正（sympy.Symbol('1') - 1）
- 行列 P が必ずしも正定値（or 負定値）にならない
- P を分割しても最適化ルーチンはまわらない
- 目的変数の数 Primal: $$J \times (D + 1) + J \times L + H$$ に対して、Dual: $$J \times L + H \times 2H + J \times (L + U) \times 2$$ ここで D は入力データの次元