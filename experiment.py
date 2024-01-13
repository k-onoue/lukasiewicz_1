import os
import time

import cvxpy as cp
import numpy as np

from src.setup_problem_dual import Setup
# from src.objective_function_2 import ObjectiveFunction
from src.objective_function import ObjectiveFunction



data_dir_path = 'inputs/pima_indian_diabetes'
train_data_dir_path = os.path.join(data_dir_path, "train")
file_list = os.listdir(train_data_dir_path)

L_files = [filename for filename in file_list 
        if filename.startswith('L') and filename.endswith('.csv')]

U_files = [filename for filename in file_list 
        if filename.startswith('U') and filename.endswith('.csv')]

file_names_dict = {
    'supervised': L_files,
    'unsupervised': U_files,
    'rule': ['rules.txt']
}



if __name__ == '__main__':
    start = time.time()

    problem_instance = Setup(train_data_dir_path, file_names_dict, ObjectiveFunction, c1=10, c2=10)

    objective, constraints = problem_instance.main()

    problem = cp.Problem(objective, constraints)
    result = problem.solve(verbose=True)

    end = time.time()

    print()
    print()
    print()
    print()
    print()
    print(f"The total time taken was {end - start}!")