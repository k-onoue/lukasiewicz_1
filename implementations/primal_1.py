import os
import pandas as pd
import numpy as np
from cvxopt import matrix
from cvxopt.solvers import qp



# load and convert data, describe problem settings, etc
data_dir_path = '../inputs/toy_data/'
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
U = np.array(df_U) # data for logcal constriant


t = 3 # number of tasks 
v = 2 # number of logical constraints considered (cardinality of KB)
ns = 0 # number of pointwise constraints to be counted later


l = [] # L_j の要素数のリスト
s = [] # S_j の要素数のリスト
S = [] # data for consistency constraints

for i in range(t):
    if v != 0:
        u = len(U)
        S_i = np.concatenate((L[i][:, :2], U), axis=0)
        S.append(S_i)
    else:
        u = 0
        S_i = L[i][:, :2]
        S.append(S_i)
    l.append(len(L[i]))
    ns += len(L[i])
    s.append(len(S_i))

S = np.stack(S)


c1 = 2.5 # degree of satisfaction for pointwise slacks
c2 = 2.5 # degree of satisfaction for logical slacks


lb = np.zeros((3 * t + ns + v, 1)) # lower bounds for variables to be optimized
ub = np.zeros((3 * t + ns + v, 1)) # upper bounds for variables to be optimized

for i in range(3 * t):
    lb[i, 0] = -100000
    ub[i, 0] = 100000

for i in range(3 * t, 3 * t + ns + v):
    lb[i, 0] = 0
    ub[i, 0] = 100000



# 目的関数の定義
# 1/2 x.T @ P @ x + q.T @ x
# の P と q を定義
# x = [w_{j1}, w_{j2}, b_j, ξ_{jl}, ξ_h] 
# len(x) = 23

H = np.zeros((3 * t + ns + v, 3 * t + ns + v))

for i in range(3 * t):
    if (i + 1) % 3 != 0:
        H[i, i] = 1


f = np.zeros((3 * t + ns + v, 1))

for i in range(3 * t + ns):
    f[i, 0] = c1

for i in range(3 * t + ns, 3 * t + ns + v, 1):
    f[i, 0] = c2




# 制約の定義
# A @ x <= b

tmp = 84

A = np.zeros((tmp, 3 * t + ns + v))
b = np.zeros((tmp, 1))
count = 0 # 制約条件の行数カウンタ


# pointwise constraints
for i in range(t):
    k = 3 * i
    for j in range(l[i]):
        A[j + count, k] = -2 * L[i][j, 0] * L[i][j, 2]
        A[j + count, k + 1] = -2 * L[i][j, 1] * L[i][j, 2]
        A[j + count, k + 2] = -2 * L[i][j, 2]
        A[j + count, count + j + 3 * t] = -2
        b[j + count, 0] = -1 - L[i][j, 2]
    count += l[i]


# consistency constraints
if v > 0:
    for i in range(t):
        k = 3 * i
        for j in range(s[i]):
            A[j + count, k] = S[i, j, 0]
            A[j + count, k + 1] = S[i, j, 1]
            A[j + count, k + 2] = 1
            b[j + count, 0] = 1
            A[j + count + s[i], k] = - S[i, j, 0]
            A[j + count + s[i], k + 1] = - S[i, j, 1]
            A[j + count + s[i], k + 2] = - 1
            b[j + count + s[i], 0] = 0

        #     print(f'j + count: {j + count}')
        #     print(f'j + count + s[i]: {j + count + s[i]}')
        # print()

        count += 2 * s[i]


# logical constraints
if v > 0:
    for i in range(v):
        k = 3 * i
        for j in range(u):
            A[j + count, k] = U[j, 0]
            A[j + count, k + 1] = U[j, 1]
            A[j + count, k + 2] = 1
            A[j + count, k + 3] = -U[j, 0]
            A[j + count, k + 4] = -U[j, 1]
            A[j + count, k + 5] = -1
            A[j + count, 3 * t + ns + i] = -1
            b[j + count, 0] = 0

            # print(f'k: {k, k+1, k+2, k+3, k+4, k+5}')
            # print(f'3*t+ns+i: {3*t+ns+i}')
            # print()

        count += u





def main():
    from cvxopt import matrix
    from cvxopt.solvers import qp


    P = matrix(H)
    q = matrix(f)
    G = matrix(A)
    h = matrix(b)

    sol = qp(P=P, q=q, G=G, h=h)

    print(sol)


if __name__ == '__main__':
    main()