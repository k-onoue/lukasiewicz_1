from typing import List, Tuple
import cvxpy as cp
import numpy as np
from .misc import timer


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
    def __init__(self, obj: Setup_, kernel_function: object = None) -> None:
        self.L = obj.L
        self.U = obj.U
        self.S = obj.S

        self.len_j = obj.len_j
        self.len_l = obj.len_l
        self.len_u = obj.len_u
        self.len_s = obj.len_s
        self.len_h = obj.len_h 
        self.len_i = obj.len_i

        self.lambda_jl  = obj.lambda_jl
        self.lambda_hi  = obj.lambda_hi
        self.eta_js     = obj.eta_js
        self.eta_hat_js = obj.eta_hat_js

        self.M = obj.M 
        self.q = obj.q

        self.predicate_names = list(obj.predicates_dict.keys())

        if kernel_function == None:
            self.k = self.linear_kernel
        else:
            self.k = kernel_function

    def linear_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.dot(x1, x2)
    

    # for j in range(self.len_j): の j は全部 0
    def _mapping_variables(self) -> Tuple[dict, List[cp.Variable]]:
        mapping_x_i_list = []
        x_list = []

        for j in range(self.len_j):
            mapping_x_i = {}
            x = []

            mapping_x_i["lambda_jl"] = {}
            for l in range(self.len_l):
                mapping_x_i['lambda_jl'][(j, l)] = len(x)
                x.append(self.lambda_jl[j, l])

            mapping_x_i["lambda_hi"] = {}
            for h in range(self.len_h):
                for i in range(self.len_i):
                    mapping_x_i["lambda_hi"][(h, i)] = len(x)
                    x.append(self.lambda_hi[h, i])

            mapping_x_i['delta_eta_js'] = {}
            for s in range(self.len_s):
                mapping_x_i["delta_eta_js"][(j, s)] = len(x)
                x.append(self.eta_js[j, s] - self.eta_hat_js[j, s])

            mapping_x_i_list.append(mapping_x_i)
            x_list.append(x)

        return mapping_x_i_list, x_list
    
    @timer
    def _construct_P_j(self, 
                       j: int, 
                       mapping_x_i: dict, 
                       x: List[cp.Variable]) -> Tuple[cp.Variable, np.ndarray]: 

        P = np.zeros((len(x), len(x)))

        key = self.predicate_names[j]
        L = self.L[key]
        U = self.U[key]
        S = self.S[key]

        for l_row in range(self.len_l):
            row = mapping_x_i['lambda_jl'][(j, l_row)]
            x_row = L[l_row, :-1]
            y_row = L[l_row, -1]
            for l_col in range(self.len_l):
                col = mapping_x_i['lambda_jl'][(j, l_col)]
                x_col = L[l_col, :-1]
                y_col = L[l_col, -1]
                P[row, col] += 4 * y_row * y_col * self.k(x_row, x_col)
            
        # for h_row in range(self.len_h):
        #     for h_col in range(self.len_h):
        #         for i_row in range(self.len_i):
        #             for i_col in range(self.len_i):
        #                 for u_row in range(self.len_u):
        #                     for u_col in range(self.len_u):
        #                         M_row = self.M[h_row][i_row, u_row]
        #                         M_col = self.M[h_col][i_col, u_col]
        #                         if M_row != 0 and M_col != 0:
        #                             row = mapping_x_i['lambda_hi'][(h_row, i_row)]
        #                             col = mapping_x_i['lambda_hi'][(h_col, i_col)]
        #                             x_row = U[u_row]
        #                             x_col = U[u_col]
        #                             P[row, col] += M_row * M_col * self.k(x_row, x_col)
                
        for h_row in range(self.len_h):
            for h_col in range(self.len_h):
                for i_row in range(self.len_i):
                    for i_col in range(self.len_i):
                        for u_row in range(self.len_u):
                            for u_col in range(self.len_u):

                                M_row = self.M[h_row][i_row, u_row]
                                M_col = self.M[h_col][i_col, u_col]
                                if M_row != 0 and M_col != 0:
                                    row = mapping_x_i['lambda_hi'][(h_row, i_row)]
                                    col = mapping_x_i['lambda_hi'][(h_col, i_col)]
                                    x_row = U[u_row]
                                    x_col = U[u_col]
                                    P[row, col] += M_row * M_col * self.k(x_row, x_col)
        

        print(self.M)


        # # Create the indices for einsum
        # h_indices = 'hijklmnopqr'[:self.len_h]
        # i_indices = 'ijklmnopqrst'[:self.len_i]
        # u_indices = 'ijklmnopqrst'[:self.len_u]

        # # Construct the einsum expression
        # expression = f"{h_indices}i,{h_indices}u,{i_indices}j,{u_indices}k,{i_indices}l,{u_indices}m,{h_indices}n,{i_indices}o,{u_indices}p,{h_indices}q,{i_indices}r,{u_indices}s->\
        #             {h_indices}{i_indices},{h_indices}{i_indices}"

        # # Perform the einsum operation
        # result = np.einsum(expression, self.M, self.M, U, U, self.M, self.M, U, U, self.M, self.M, U, U, self.k(U[:, None], U[None, :]))

        # # Update the P matrix with the result
        # P += result



        K = np.zeros(shape=(self.len_u, self.len_u))
        for u in range(self.len_u):
            for u_dash in range(self.len_u):
                x = U[u]
                x_dash = U[u_dash]

                K[u, u_dash] = self.k(x, x_dash)

        for h_row in range(self.len_h):
            for h_col in range(self.len_h):
                for i_row in range(self.len_i):
                    for i_col in range(self.len_i):
                        M_row = self.M[h_row][i_row, :self.len_u]
                        M_col = self.M[h_col][i_col, :self.len_u]

                        print(M_row.shape)
                        print(K.shape)


                        row = mapping_x_i['lambda_hi'][(h_row, i_row)]
                        col = mapping_x_i['lambda_jl'][(h_col, i_col)]
                        P[row, col] += np.einsum('i,j,ij', M_row, M_col, K)

        for s_row in range(self.len_s):
            row = mapping_x_i["delta_eta_js"][j, s_row]
            x_row = S[s_row]
            for s_col in range(self.len_s):
                col = mapping_x_i["delta_eta_js"][j, s_col]
                x_col = S[s_col]
                P[row, col] += self.k(x_row, x_col)
                
        for l in range(self.len_l):
            row = mapping_x_i["lambda_jl"][(j, l)]
            x_l = L[l, :-1]
            y_l = L[l, -1]
            for h in range(self.len_h):
                for i in range(self.len_i):
                    col = mapping_x_i["lambda_hi"][(h, i)]
                    for u in range(self.len_u):
                        x_u = U[u]
                        M = self.M[h][i, u]
                        if M != 0:
                            P[row, col] += -4 * y_l * M * self.k(x_l, x_u)
            
        for l in range(self.len_l):
            row = mapping_x_i["lambda_jl"][(j, l)]
            x_l = L[l, :-1]
            y_l = L[l, -1]
            for s in range(self.len_s):
                col = mapping_x_i["delta_eta_js"][(j, s)]
                x_s = S[s]
                P[row, col] += 4 * y_l * self.k(x_l, x_s)
                
        for h in range(self.len_h):
            for i in range(self.len_i):
                row = mapping_x_i["lambda_hi"][(h, i)]
                for s in range(self.len_s):
                    col = mapping_x_i["delta_eta_js"][(j, s)]
                    x_s = S[s]
                    for u in range(self.len_u):
                        x_u = U[u]
                        M = self.M[h][i, u]
                        if M != 0:
                            P[row, col] += -2 * M * self.k(x_u, x_s)
        
        # P = (-1/2)*(P+P.T)/2
        P = (P+P.T)/2
        return cp.vstack(x), P
    
    # def _construct_q_(self, mapping_x_i: Dict[str, dict]) -> np.ndarray:
    #     q_ = np.zeros((len(x), 1))
    #     col = 0

    #     for j in range(self.len_j):
    #         for l in range(self.len_l):
    #             row = mapping_x_i["lambda_jl"][(j, l)]
    #             q_[row, col] += 1

    #             print(row)
        
    #     for h in range(self.len_h):
    #         for i in range(self.len_i):
    #             row = mapping_x_i["lambda_hi"][(h, i)]
    #             for u in range(self.len_u):
    #                 M = self.M[h][i, u]
    #                 q = self.q[h][i, 0]
    #                 q_[row, col] += (1/2) * M + q

    #             print(row)
        
    #     for j in range(self.len_j):
    #         for s in range(self.len_s):
                 

    def construct(self) -> cp.Expression:

        objective_function = 0
        mapping_x_i_list, x_list = self._mapping_variables()

        for j, (mapping_x_i, x) in enumerate(zip(mapping_x_i_list, x_list)):
            x, P = self._construct_P_j(j, mapping_x_i, x)
            objective_function = (-1/2) * cp.quad_form(x, P)

        for j in range(self.len_j):
            for l in range(self.len_l):
                objective_function += self.lambda_jl[j, l]
        
        for h in range(self.len_h):
            for i in range(self.len_i):
                for u in range(self.len_u):
                    objective_function += self.lambda_hi[h, i] * (1/2 * self.M[h][i, u] + self.q[h][i, 0])
            
        for j in range(self.len_j):
            for s in range(self.len_s):
                objective_function += (-1/2) * (self.eta_js[j, s] + self.eta_hat_js[j, s])

        objective_function = cp.Maximize(objective_function)

        return objective_function
    

