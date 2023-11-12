# 概要

[Learning Lukasiewicz Logic Fragments by Quadratic Programming](http://ecmlpkdd2017.ijs.si/papers/paperID223.pdf) のアルゴリズムの実装です．


## To do
- logical constraints の処理の説明の完成
- winston benchmark の notebook の整理

- 画像を 128 x 128 にリサイズ
- PCA で特徴量を 200 次元に落とす

- PCA する前に 0 ~ 1 に値を収めたほうが良かったかも

- accuracy_score, classification_report 



## 環境構築

```{shell}
$ python3 -m venv myenv
$ source myenv/bin/activate
$ pip install -r requirements.txt
```

## ディレクトリ構成

- lukasiewicz_1/ ... ホームディレクトリ
  - images/ ... ドキュメント用画像ファイル
  - inputs/ ... 入力データ
    - toy_data/ ... Toy problem の入力データ
    - winston_data/ ... Winston benchmark の入力データ
  - notebooks/ ... 実験用 Jupyter Notebook
    - toy_problem/ 
    - winston/ 
      
  - src/ ... ソースコード
    - setup_problem.py ... cvxpy.Problem に引数として渡す目的関数と制約を生成する
    - operators.py ... 制約を記述する際に使用される論理演算子を意味付けする
    - process_fol.py ... .txt ファイルとして受け取った一階述語論理式の集合を，最終的に各論理式を "⊕" の式に変換し，論理式のリストを作成する
    - misc.py ... 様々なユーティリティ関数


## 一階述語論理で書かれた制約の処理の流れ

制約 (constraints) は，

- Pointwise constraints
- Logical constrants
- Consistency constrants

の 3 種類からなり，ここでは 2 の Logical constraints の処理の流れを紹介する．

### 1. 論理式の受け取り

まず， Knowledge Base (以降 KB とする) と呼ばれる一階述語論理式の集合を .txt ファイルとして受け取る．論理式は formula と呼ばれ， $φ$ と表される．下は KB の例であり，4 つの formula，つまり $φ_1, φ_2, ..., φ_4$ からなる．

```
hair(x) → mammal(x) 
carnivore(x) ⊗ tawny(x) ⊗ darkspots(x) → cheetah(x)
mammal(x) ⊕ bird(x)
white(x) → ¬ black(x)
```

ここで，全ての論理式は先頭の全称量化子 "∀x" が省略されているものと考える．つまり実際は

```
∀x(hair(x) → mammal(x)) 
```

となっているものとする．

ちなみに含まれる全ての式は， $x_1, \dots, x_m, y$ を命題変数とすると

$$(x_1 ⊗ \cdots ⊗ x_m) → y$$

の形 (Horn clause) になっている．

### 2. 論理式の不等式への変換

制約不等式は以下のように表される．

$$¬ f_φ(\boldsymbol{\bar p}) = 1 - f_φ(\boldsymbol{\bar p}) \leq \xi$$

ここで， $f_φ$ は formula $φ$ と一対一に対応する，定義域が $[0, 1]^n$ ， 値域が $[0, 1]$ の関数であり，
$f_φ(\boldsymbol{\bar{p}})$ は $φ$ に含まれる全ての predicate を，その predicate が取りうる全てのデータ点で評価をしたものという意味である．

上記の KB の 1 つ目の論理式 $φ: hair(x) → mammal(x)$ を例に出して説明する．

$hair(x)$， $mammal(x)$ ともに同じデータ点 $x \in U$ で評価されるとすると

$$
\begin{align*}
\forall x (hair(x) → mammal(x))
&\simeq \wedge_{x \in U} (hair(x) → mammal(x)) 
\end{align*}
$$

と変換され，この右辺を $φ'$ とおく．$U = \{x_1, x_2, \dots, x_n\}$ とすると，制約不等式 $¬ f_{φ'}(\boldsymbol{\bar p}) = 1 - f_{φ'}(\boldsymbol{\bar p}) \leq \xi$

$$
\begin{align*}
&¬ \left\{(hair(x_1) → mammal(x_1)) \wedge \cdots \wedge (hair(x_n) → mammal(x_n)) \right\} \leq \xi \\
\iff &¬ (hair(x_1) → mammal(x_1)) \vee \cdots \vee ¬ (hair(x_n) → mammal(x_n))  \leq \xi
\end{align*}
$$

となる．これは各 $¬ (hair(x) → mammal(x))$ が個別に $\xi$ 以下であることと等しいので，制約不等式は最終的に以下のように $n$ 本の不等式に分解される．


$$
\begin{align*}
&¬ (hair(x_1) → mammal(x_1)) \leq \xi \\
&¬ (hair(x_2) → mammal(x_2)) \leq \xi \\
&\cdots \\
&¬ (hair(x_n) → mammal(x_n)) \leq \xi
\end{align*}
$$




  
## 実験コードの実行手順

1. ライブラリのインポート

```
import cvxpy as cp

from src.setup_problem import Setup
```

2. 入力データの準備

```
# 例
data_dir_path = "./inputs/toy_data"

# 例
file_names_dict = {
    "supervised": ["L_p1(x)", "L_p2(x)", "L_p3(x)"],
    "unsupervised": ["U"]
    "rule": ["rules"]
}
```

3. 目的関数，制約の構成

```
problem_instance = Setup(data_dir_path, file_names_dict)

# 例
objective, constraints = problem_instance.main(c1=10, c2=10)
```

4. 最適化の実行

```
problem = cp.Problem(objective, constraints)
result = problem.solve(verbose=True)
```

5. 評価

```
from src.misc import test_trained_predicate

p_dict = problem_instance.predicates_dict
test_data = problem_instance.L # ここでは学習に使用した教師ありデータを使用しているが，正解ラベルが与えられていれば何でも良い

res_dict = test_trained_predicate(p_dict, test_data)
```

6. 可視化（データの次元が 2 の時のみ）

```
from src.misc import visualize_result

visualize_result(problem_instance)
```

![Toy problem visualization](./images/toy_vis.png)


