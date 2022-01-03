import xlwings as xw
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import numpy as np

def 多因素方差分析显著性检验():
    肾病病例工作表 = xw.Book('肾穿病例整理原始数据_删除了一些有空值的数据_副本.xlsx').sheets(1)

    # 读数据的时候，数据的索引是从矩形区域的左上角到矩形区域的右下角，第一个是左上角坐标，第二个是右下角坐标
    糖肾标签 = np.array(肾病病例工作表.range((2, 6), (152, 6)).value)  # 分三类：无糖肾、糖肾、二者合并。
    糖肾标签.resize(151, 1)
    # print(糖肾标签)
    # 糖肾标签 = np.resize(糖肾标签, (151, 1))
    原始糖肾数据 = np.array(肾病病例工作表.range((2, 30), (152, 50)).value)  # 这个是没拼接标签的数据
    拼接标签的原始糖肾数据 = np.hstack((原始糖肾数据, 糖肾标签))
    pd形式数据 = pd.DataFrame(拼接标签的原始糖肾数据,
                          columns=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13',
                                   'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'label'])
    # print('\n', pd形式数据)

    # 多因素方差分析，不考虑各个因素之间的相互作用
    # '''
    不考虑因素之间的相互作用 = 'label ~ A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 + A9 + A10 + A11' \
                   ' + A12 + A13 + A14 + A15 + A16 + A17 + A18 + A19 + A20 + A21'
    model = ols(不考虑因素之间的相互作用, pd形式数据).fit()
    anovat = anova_lm(model)
    print("不考虑因素之间的影响",'\n' , anovat)
    # '''

    # 考虑几个重要特征之间的相互影响
    '''
    考虑重要因素之间的相互作用 = 'label ~ A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 + A9 + A10 + A11 + A12' \
                    ' + A13 + A14 + A15 + A16 + A17 + A18 + A19 + A20 + A21 + A9 * A14 * A15 * A12 * A17 * A11 * A1'
    # + A9 * A14 * A15 * A12 * A17 * A11 * A1 * A16
    考虑重要因素之间的相互作用 = 'label ~ A9 * A14 * A15 * A12 * A17 * A11 * A1'
    model = ols(考虑重要因素之间的相互作用, pd形式数据).fit()
    anovat = anova_lm(model)
    print("考虑重要因素之间的影响", '\n', anovat)
    '''

    ''' 数据精度过高造成内存不足，无法计算考虑因素之间相互影响时的结果
    # 多因素方差分析，考虑各个因素之间的相互作用
    考虑因素之间的相互作用 = 'label ~ A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 + A9 + A10 + A11 + A12 + A13 + A14 + A15 + A16 + A17 + A18 + A19 + A20 + A21 + A1 * A2* A3 * A4 * A5 * A6 * A7 * A8 * A9 * A10 * A11 * A12 * A13 * A14 * A15 * A16 * A17 * A18 * A19 * A20 * A21'
    model = ols(考虑因素之间的相互作用, pd形式数据).fit()
    anovat = anova_lm(model)
    print("考虑因素之间的影响",'\n' , anovat)
    '''





























# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    多因素方差分析显著性检验()
