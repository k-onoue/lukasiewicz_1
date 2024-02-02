from typing import List, Tuple
import cvxpy as cp
import numpy as np
from .misc import timer
# from .misc import get_near_nsd_matrix
from .misc import get_near_psd_matrix


# from .setup_problem import Setup
class Setup_:
    """
    型ヒント用（circular import の回避のため）
    """
    # def __init__(self):
    #     pass
    
    # FOLConverter の簡易動作確認用
    def __init__(self):
        self.len_j = 3
        self.len_u = 6



class ObjectiveFunction:
    def __init__(self, 
                 obj: Setup_, 
                 target_predicate_name: str,
                 kernel_function: object = None) -> None:
        
        predicate_names = list(obj.predicates_dict.keys())
        self.target_p_name = target_predicate_name
        self.target_p_idx = predicate_names.index(self.target_p_name)

        self.L = obj.L
        self.U = obj.U
        self.S = obj.S

        # self.len_j = obj.len_j
        self.len_l = obj.len_l
        self.len_u = obj.len_u
        self.len_s = obj.len_s
        self.len_h = obj.len_h 
        self.len_i = obj.len_i

        self.lambda_jl  = obj.lambda_jl[self.target_p_idx]
        self.lambda_hi  = obj.lambda_hi
        self.eta_js     = obj.eta_js[self.target_p_idx]
        self.eta_hat_js = obj.eta_hat_js[self.target_p_idx]

        self.M = obj.M 
        self.q = obj.q

        if kernel_function == None:
            self.k = self.linear_kernel
        else:
            self.k = kernel_function

    def linear_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.dot(x1, x2)
    
    def compute_kernel_matrix(self, X1, X2):
        """
        Compute the kernel matrix between two matrices.

        Parameters:
        - X1: First input matrix (n x m)
        - X2: Second input matrix (n x m)
        - kernel_function: Kernel function to use (default is dot product)

        Returns:
        - Kernel matrix (n x n)
        """
        kernel_function = self.k

        n1, m1 = X1.shape
        n2, m2 = X2.shape

        if m1 != m2:
            raise ValueError("Input matrices must have the same number of features (columns)")

        K_matrix = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                K_matrix[i, j] = kernel_function(X1[i, :], X2[j, :])

        return K_matrix
    
    def _mapping_variables(self) -> Tuple[dict, List[cp.Variable]]:
        mapping_x_i = {}
        x = []

        mapping_x_i["lambda_jl"] = {}
        for l in range(self.len_l):
            mapping_x_i['lambda_jl'][(0, l)] = len(x)
            x.append(self.lambda_jl[l])

        mapping_x_i["lambda_hi"] = {}
        for h in range(self.len_h):
            for i in range(self.len_i):
                mapping_x_i["lambda_hi"][(h, i)] = len(x)
                x.append(self.lambda_hi[h, i])

        mapping_x_i['delta_eta_js'] = {}
        for s in range(self.len_s):
            mapping_x_i["delta_eta_js"][(0, s)] = len(x)
            x.append(self.eta_js[s] - self.eta_hat_js[s])

        return mapping_x_i, x
    
    @timer
    def _construct_P_j(self,
                       mapping_x_i: dict,
                       x: List[cp.Variable]) -> Tuple[cp.Variable, np.ndarray]: 

        P = np.zeros((len(x), len(x)))

        ############################################
        ############################################
        ############################################
        print(f'shape of P: {P.shape}')
        ############################################
        ############################################
        ############################################

        L = self.L[self.target_p_name]
        U = self.U[self.target_p_name]
        S = self.S[self.target_p_name]

        start_col = self.target_p_idx * self.len_u
        end_col = start_col + self.len_u
        M = [M_h[:, start_col:end_col] for M_h in self.M]

        # P_{11}
        for l in range(self.len_l):
            for l_ in range(self.len_l):
                row = mapping_x_i['lambda_jl'][(0, l)]
                col = mapping_x_i['lambda_jl'][(0, l_)]

                x_l  = L[l, :-1]
                x_l_ = L[l_, :-1]
                k    = self.k(x_l, x_l_)

                y_l  = L[l, -1]
                y_l_ = L[l_, -1]

                P[row, col] += 4 * y_l * y_l_ * k


        ############################################
        ############################################
        print('finish l')
        ############################################
        ############################################


        # P_{22} using einsum
        K = self.compute_kernel_matrix(U, U)
        print(K.shape)

        for h in range(self.len_h):
            for h_ in range(self.len_h):
                for i in range(self.len_i):
                    for i_ in range(self.len_i):
                        row = mapping_x_i['lambda_hi'][(h, i)]
                        col = mapping_x_i['lambda_hi'][(h_, i_)]

                        m  = M[h][i, :]
                        m_ = M[h_][i_, :]

                        P[row, col] += np.einsum("i,j,ij", m, m_, K)

        ############################################
        ############################################
        print('finish h')
        ############################################
        ############################################

        # P_{33}
        for s in range(self.len_s):
            for s_ in range(self.len_s):
                row = mapping_x_i['delta_eta_js'][(0, s)]
                col = mapping_x_i['delta_eta_js'][(0, s_)]

                x_s  = S[s]
                x_s_ = S[s_]
                k  = self.k(x_s, x_s_)

                P[row, col] += k

        ############################################
        ############################################
        print('finish s')
        ############################################
        ############################################

        # P_{12}
        for l in range(self.len_l):
            for h in range(self.len_h):
                for i in range(self.len_i):
                    row = mapping_x_i['lambda_jl'][(0, l)]
                    col = mapping_x_i['lambda_hi'][(h, i)]

                    y_l = L[l, -1] 

                    for u in range(self.len_u):
                        m = M[h][i, u]

                        x_l = L[l, :-1]
                        x_u = U[u]
                        k   = self.k(x_l, x_u)

                        P[row, col] += (-4) * y_l * m * k

        ############################################
        ############################################
        print('finish l h')
        ############################################
        ############################################

        # P_{13}
        for l in range(self.len_l):
            for s in range(self.len_s):
                row = mapping_x_i['lambda_jl'][(0, l)]
                col = mapping_x_i['delta_eta_js'][(0, s)]

                y_l = L[l, -1]

                x_l = L[l, :-1]
                x_s = S[s]
                k = self.k(x_l, x_s)

                P[row, col] += 4 * y_l * k
        

        ############################################
        ############################################
        print('finish l s')
        ############################################
        ############################################


        # P_{23}
        for h in range(self.len_h):
            for i in range(self.len_i):
                for s in range(self.len_s):
                    for u in range(self.len_u):
                        row = mapping_x_i['lambda_hi'][(h, i)]
                        col = mapping_x_i['delta_eta_js'][(0, s)]

                        m = M[h][i, u]

                        x_u = U[u]
                        x_s = S[s]
                        k = self.k(x_u, x_s)

                        P[row, col] += (-2) * m * k

        ############################################
        ############################################
        print('finish h s')
        ############################################
        ############################################



        P = (P+P.T)/2
        return cp.vstack(x), P
                 

    def construct(self) -> cp.Expression:

        objective_function = 0

        mapping_x_i, x = self._mapping_variables()
        x, P = self._construct_P_j(mapping_x_i, x)

        # 計算安定性のため
        P += np.diag(np.ones(P.shape[0])) * 1e-6

        objective_function = (-1/2) * cp.quad_form(x, P)

        for l in range(self.len_l):
            objective_function += self.lambda_jl[l]
        
        for h in range(self.len_h):
            for i in range(self.len_i):
                objective_function += self.lambda_hi[h, i] * (1/2 * self.M[h][i, :].sum() + self.q[h][i, 0])

        for s in range(self.len_s):
            objective_function += (-1/2) * (self.eta_js[s] + self.eta_hat_js[s])

        objective_function = cp.Maximize(objective_function)

        return objective_function