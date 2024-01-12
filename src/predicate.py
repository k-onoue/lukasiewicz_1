import numpy as np


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


class Predicate_dual:
    def __init__(self, obj: Setup_, name: str, kernel_function: object = None) -> None:
        self.c1 = obj.c1
        self.c2 = obj.c2

        self.L = obj.L[name]
        self.U = obj.U[name]
        self.S = obj.S[name]

        self.len_l = obj.len_l
        self.len_u = obj.len_u
        self.len_s = obj.len_s
        self.len_h = obj.len_h 
        self.len_i = obj.len_i

        self.p_idx = list(obj.predicates_dict.keys()).index(name)

        self.lambda_jl  = obj.lambda_jl[self.p_idx, :].value
        self.lambda_hi  = obj.lambda_hi.value
        self.eta_js     = obj.eta_js[self.p_idx, :].value
        self.eta_hat_js = obj.eta_hat_js[self.p_idx, :].value

        start_col = self.p_idx * self.len_u
        end_col = start_col + self.len_u
        # self.M_j = [M_h[:, start_col:end_col] for M_h in obj.M]
        self.M_j = obj.M

        if kernel_function == None:
            self.k = self.linear_kernel
        else:
            self.k = kernel_function

        ##########################################################
        ##########################################################
        ##########################################################
        """ 
        なぜか b += 0.5 するとうまく識別できている．（p の予測値に対して 0.5 が境界）
        入力データで y in {-1, 1} であるからか
        """
        self.b = self._b() + 0.5
        # self.b = self._b()

        self.w_linear_kernel = self._w_linear_kernel() 

        self.coeff = np.append(self.w_linear_kernel, self.b)


    def linear_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.dot(x1, x2)
    
    def w_dot_phi(self, x_pred: np.ndarray) -> float:
        k = self.k
        value = 0

        for l in range(self.len_l):
            x = self.L[l, :-1]
            y = self.L[l, -1]
            lmbda = self.lambda_jl[l]
            value += 2 * lmbda * y * k(x, x_pred)

        for h in range(self.len_h):
            for i in range(self.len_i):
                lmbda = self.lambda_hi[h, i]
                for u in range(self.len_u):
                    x = self.U[u]
                    M = self.M_j[h][i, u]
                    value += - lmbda * M * k(x, x_pred)
        
        for s in range(self.len_s):
            x = self.S[s]
            eta = self.eta_js[s]
            eta_hat = self.eta_hat_js[s]
            value += (eta - eta_hat) * k(x, x_pred)

        return value
    
    def _w_linear_kernel(self) -> np.ndarray:
        input_dim = len(self.L[0, :-1])
        w_linear_kernel = np.zeros(input_dim)

        for l in range(self.len_l):
            x = self.L[l, :-1]
            y = self.L[l, -1]
            lmbda = self.lambda_jl[l]
            w_linear_kernel += 2 * lmbda * y * x

        for h in range(self.len_h):
            for i in range(self.len_i):
                lmbda = self.lambda_hi[h, i]
                for u in range(self.len_u):
                    x = self.U[u]
                    M = self.M_j[h][i, u]
                    w_linear_kernel += - lmbda * M * x
        
        for s in range(self.len_s):
            x = self.S[s]
            eta = self.eta_js[s]
            eta_hat = self.eta_hat_js[s]
            w_linear_kernel += (eta - eta_hat) * x

        return w_linear_kernel
    
    def _b(self) -> float:
        value_tmp = 0
        count = 0

        for l in range(self.len_l):
            lmbda = self.lambda_jl[l]

            if lmbda >= 0 and lmbda <= self.c1:
                x = self.L[l, :-1]
                y = self.L[l, -1] 

                value_tmp += y - self.w_dot_phi(x)
                count += 1

        value = value_tmp / count

        return value

    def __call__(self, x_pred: np.ndarray) -> float:
        
        value = self.w_dot_phi(x_pred) + self.b

        return value