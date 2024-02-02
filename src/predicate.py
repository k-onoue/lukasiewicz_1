import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import auc
import sklearn.metrics

from sklearn.metrics import f1_score
from scipy.optimize import minimize

import optuna

def pr_auc_score():
    pass


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
    def __init__(
            self, 
            obj: Setup_, 
            name: str, 
            kernel_function: object = None,
            metrics: sklearn.metrics = None,
            opt_iter_num: int = 100
        ) -> None:
    
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
        self.M_j = [M_h[:, start_col:end_col] for M_h in obj.M]

        if kernel_function == None:
            self.k = self.linear_kernel
        else:
            self.k = kernel_function

        if metrics == None:
            self.metrics = f1_score
        else:
            self.metrics = metrics

        self.n_trials = opt_iter_num
        

        ##########################################################
        ##########################################################
        ##########################################################

        # self.b = self._b()
        self.b = self.optimize_b()

        self.w_linear_kernel = self._w_linear_kernel() 

        self.coeff = np.append(self.w_linear_kernel, self.b)


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

        K_matrix = kernel_function(X1, X2.T)
        return K_matrix
    
    def w_dot_phi(self, x_pred: np.ndarray) -> float:
        values = np.zeros(len(x_pred))
        
        x = self.L[:, :-1]
        y = self.L[:, -1]
        lmbda = self.lambda_jl
        K = self.compute_kernel_matrix(x, x_pred)

        values += 2 * (y * lmbda).T @ K 

        x = self.U
        lmbda = self.lambda_hi.flatten().reshape(-1, 1)
        M = np.vstack(self.M_j)
        K = self.compute_kernel_matrix(x, x_pred)

        values += (-1) * (lmbda * M @ K).sum(axis=0)

        x = self.S
        eta = self.eta_js
        eta_hat = self.eta_hat_js
        K = self.compute_kernel_matrix(x, x_pred)

        values += (eta - eta_hat) @ K
        
        return values
    
    
    def _w_linear_kernel(self) -> np.ndarray:
        """
        This should be rewritten using matrix calculations.
        """
        
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
    
    def _calculate_initial_b(self) -> float:
        x = self.L[:, :-1]
        y = self.L[:, -1]

        initial_b = np.mean(y - self.w_dot_phi(x))

        return initial_b
    
    def _calculate_score_at_b(
            self, 
            b: float,
            X: np.ndarray,
            y: np.ndarray
        ) -> float:

        y_pred = self.w_dot_phi(X)

        y_pred = np.array(y_pred) + b
        y_pred_interpreted = np.where(y_pred >= 0.5, 1, -1)

        score = self.metrics(y, y_pred_interpreted)

        return score
    
    def optimize_b(self):
        initial_b = self._calculate_initial_b()

        X_train = self.L[:, :-1]
        y_train = self.L[:, -1]

        range_size = 5

        min_bound = initial_b - 0.5 * range_size
        max_bound = initial_b + 0.5 * range_size
        
        print()
        print(f'min_bound: {min_bound}')
        print(f'max_bound: {max_bound}')
        print()

        def objective(trial):
            b = trial.suggest_float('b', min_bound, max_bound)
            return self._calculate_score_at_b(b, X_train, y_train)  
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        study.enqueue_trial(initial_b)

        optimal_b = study.best_params['b']
        return optimal_b


    def __call__(self, x_pred: np.ndarray) -> float:
        
        value = self.w_dot_phi(x_pred) + self.b

        return value