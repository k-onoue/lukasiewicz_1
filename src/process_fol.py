from .operators import negation
from .operators import Semantisize_symbols

# symbols_1 = ['¬', '∧', '∨', '⊗', '⊕', '→']

symbols_tmp = Semantisize_symbols()
symbols_1_semanticized = symbols_tmp.symbols_1_semanticized
symbols_3_semanticized = symbols_tmp.symbols_3_semanticized
symbols = list(symbols_1_semanticized.keys()) + list(symbols_3_semanticized.keys())


class FOLConverter:
    """
    Knowledge base (KB) を .txt ファイルで受け取り，
    その中の述語論理で記述された rule を
    '¬' と '⊕' だけの形に変換し，
    リストとして返す
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.KB = self._construct_KB()
        self.new_KB = None
        
    def _construct_KB(self):
        """
        KB の .txt ファイルを読み込む
        """    
        KB = []

        with open(self.file_path, 'r') as file:
            for line in file:
                KB.append(line.split())

        return KB     

    def _check_implication(self, formula):
        """
        formula (リスト) について，
        含意記号 '→' の数を調べ，
        その数が 1 以下になっているかを確認する
        """
        
        # 実質，ここには 1 つのインデックスしか入らない
        implication_idxs = []

        for i, item in enumerate(formula):
            if item == '→':
                implication_idxs.append(i)
        
        implication_num = len(implication_idxs)
        
        if implication_num == 0:
            return False, None
        elif implication_num == 1:
            implication_idx = implication_idxs[0]
            return True, implication_idx
        else:
            print('this formula may be invalid')

    def _eliminate_implication(self, formula):
        """
        formula (リスト) 内に含意記号 '→' あれば変換し，消去する 
        """
        implication_flag, target_idx = self._check_implication(formula)

        if implication_flag:
            # 含意記号 '→' を境に formula (list) を 2 つに分ける
            x = formula[:target_idx]
            y = formula[target_idx + 1:]

            # x → y = ¬ x ⊕ y
            x_new = negation(x)
            y_new = y
            new_operator = ['⊕']

            new_formula = x_new + new_operator + y_new
        else:
            new_formula = formula

        return new_formula
    

    def main(self):
        """
        新しい KB を返す
        """
        new_KB = []

        for formula in self.KB:
            new_formula = self._eliminate_implication(formula)
            new_KB.append(new_formula)

        self.new_KB = new_KB
        
        return new_KB
    

    # # this method must be executed after executing _eliminate_implication
    # def _eliminate_o_plus(self, formula):
    #     tmp_items = []
    #     tmp_formulas = []
    #     target_idxs = [i for i, item in enumerate(formula) if item == '⊕']
    #     start_idx = 0

    #     for idx in target_idxs:
    #         tmp_formulas.append(formula[start_idx:idx])
    #         start_idx = idx + 1
        
    #     tmp_formulas.append(formula[start_idx:])

    #     for formula in tmp_formulas:
    #         for item in formula:
    #             tmp_items.append(item)

    #         tmp_items.append('+')

    #     tmp_items.pop()
        
    #     new_formula = [1] + ['∧'] + tmp_items

    #     return new_formula

    
    # def main_v2(self):
    #     new_KB = []

    #     for formula in self.KB:
    #         tmp = self._eliminate_implication(formula)
    #         new_formula = self._eliminate_o_plus(tmp)
    #         new_KB.append(new_formula)

    #     self.new_KB = new_KB

    #     return new_KB

