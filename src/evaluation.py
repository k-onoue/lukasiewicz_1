from typing import Dict, Any, List
import json

# from .setup_problem import Setup
class Setup_:
    """
    型ヒント用（circular import の回避のため）
    """
    def __init__(self):
        pass

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from .misc import is_symbol
from .operators import negation










class EvaluateModel:
    """
    学習済みモデルの評価値を算出する.
    現状 Pima Indian Diabetes dataset にのみ機能する（少しの変更を評価対象の predicate 名に変更すればおそらくうまくいく）．
    または使用するデータセットすべてにおいて，正解ラベルの列の列名を Outcome に変更すると問題なく動くかもしれない．
    """
    def __init__(self,
                 obj: Setup_,
                 test_data_path: str,
                 note: str = None) -> None:
        
        self.obj = obj
        self.test_data_path = test_data_path

        self.result_dict = {
            'name': obj.name,
            'note': note,
            'Accuracy'        : None,
            'Confusion_matrix': None,
            'Precision'       : None,
            'Recall'          : None,
            'F1-score'        : None,
            'Auc'             : None,
            'Classification_report': None,
            'VN_pointwise'  : None,
            'VN_logical'    : None,
            'VN_consistency': None,
            'VN_all'        : None,
            'VR_pointwise'  : None,
            'VR_logical'    : None,
            'VR_consistency': None,
            'VR_all'        : None
        }

    def _judge_violations_pointwise(self) -> List[bool]:

        judgements_pointwise = []

        for j, (p_name, p) in enumerate(self.obj.predicates_dict.items()):
            for l in range(self.obj.len_l):
                x = self.obj.L[p_name][l, :-1]
                y = self.obj.L[p_name][l, -1]
                xi = self.obj.xi_jl[j, l]

                if xi.value is None:
                    xi = 0
                else:
                    xi = xi.value

                LHS = (y * (2 * p(x) - 1)).value
                RHS = 1 - 2 * xi

                judgements_pointwise += [
                    LHS >= RHS
                ]
        
        return judgements_pointwise

    def _judge_violations_logical(self) -> List[bool]:
        
        judgements_logical = []

        # 仮
        U = next(iter(self.obj.U.values()))

        for u in U:
            KB_tmp = self.obj._calc_KB_at_datum(self.obj.KB, u)           

            for h, formula in enumerate(KB_tmp):
          
                xi = self.obj.xi_h[h, 0]
                if xi.value is None:
                    xi = 0
                else:
                    xi = xi.value

                formula_tmp = 0
                for item in formula:
                    if not is_symbol(item):
                        formula_tmp += item
                formula_tmp = negation(formula_tmp)
                formula_tmp = formula_tmp.value


                judgements_logical += [
                    0 <= xi,
                    formula_tmp <= xi
                ]
        
        return judgements_logical

    def _judge_violations_consistency(self) -> List[bool]:
        
        judgements_consistency = []
        for (p_name, p) in self.obj.predicates_dict.items():
            for s in range(self.obj.len_s):
                x = self.obj.S[p_name][s]

                p_x = p(x).value

                judgements_consistency += [
                    p_x >= 0,
                    p_x <= 1
                ]
        
        return judgements_consistency

    def calculate_violation(self) -> None:
        judgements_pointwise = self._judge_violations_pointwise()
        judgements_logical = self._judge_violations_logical()
        judgements_consistency = self._judge_violations_consistency()
        judgements = judgements_pointwise + judgements_logical + judgements_consistency

        violations_info = {
            'pointwise': judgements_pointwise,
            'logical': judgements_logical,
            'consistency': judgements_consistency,
            'all': judgements
        }

        for key, value in violations_info.items():
            key = 'VN' + '_' + key
            self.result_dict[key] = float(len(value) - sum(value))

        for key, value in violations_info.items():
            key = 'VR' + '_' + key
            self.result_dict[key] = float((len(value) - sum(value)) / len(value))

    def calculate_basic_scores(self) -> None:
        test_df = pd.read_csv(self.test_data_path, index_col=0)
        test_data = {
            'Outcome': np.array(test_df)
        }

        p_dict = self.obj.predicates_dict
        selected_predicates = ['Outcome']
        # selected_p_dict = {key: value for key, value in p_dict.items() if key in selected_predicates}

        X_test = test_data['Outcome'][:, :-1]
        y_test = test_data['Outcome'][:, -1]

        y_pred = p_dict['Outcome'](X_test).value

        y_pred_interpreted = np.where(y_pred >= 0.5, 1, -1)

        accuracy = accuracy_score(y_test, y_pred_interpreted)
        conf_matrix = confusion_matrix(y_test, y_pred_interpreted)
        precision = precision_score(y_test, y_pred_interpreted)
        recall = recall_score(y_test, y_pred_interpreted)
        f1 = f1_score(y_test, y_pred_interpreted)
        class_report = classification_report(y_test, y_pred_interpreted)
        roc_auc = roc_auc_score(y_test, y_pred)

        self.result_dict['Accuracy'] = float(accuracy)
        self.result_dict['Confusion_matrix'] = conf_matrix.tolist()
        self.result_dict['Precision'] = float(precision)
        self.result_dict['Recall'] = float(recall)
        self.result_dict['F1-score'] = float(f1)
        self.result_dict['Classification_report'] = class_report
        self.result_dict['Auc'] = float(roc_auc)

    def save_result_as_json(self, file_path) -> None:
        with open(file_path, 'w') as f:
            json.dump(self.result_dict, f, indent=4)

    def __call__(self, save_file_path: str ='./result_formatted.json') -> None:
        self.calculate_basic_scores()
        self.calculate_violation()
        self.save_result_as_json(file_path=save_file_path)
