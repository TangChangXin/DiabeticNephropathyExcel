import xlwings as xw
from sklearn import svm
from sklearn import preprocessing
from sklearn import metrics
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


def 参数搜索(工作表):
    # 读数据的时候，数据的索引是从矩形区域的左上角到矩形区域的右下角，第一个是左上角坐标，第二个是右下角坐标
    OCTA糖肾标签 = np.array(工作表.range((5, 5), (49, 5)).value)  # 分两类
    OCTA糖肾标签.resize(45, 1)

    特征1 = np.resize(np.array(工作表.range((5, 56), (49, 56)).value), (45, 1))  # 这个是没拼接标签的数据
    特征2 = np.resize(np.array(工作表.range((5, 57), (49, 57)).value), (45, 1))  # 这个是没拼接标签的数据
    特征3 = np.resize(np.array(工作表.range((5, 75), (49, 75)).value), (45, 1))  # 这个是没拼接标签的数据
    特征4 = np.resize(np.array(工作表.range((5, 80), (49, 80)).value), (45, 1))  # 这个是没拼接标签的数据
    特征5 = np.resize(np.array(工作表.range((5, 88), (49, 88)).value), (45, 1))  # 这个是没拼接标签的数据
    特征6 = np.resize(np.array(工作表.range((5, 90), (49, 90)).value), (45, 1))  # 这个是没拼接标签的数据
    # 特征7 = np.resize(np.array(工作表.range((5, 141), (49, 141)).value), (45, 1))  # 这个是没拼接标签的数据
    # 特征8 = np.resize(np.array(工作表.range((5, 147), (49, 147)).value), (45, 1))  # 这个是没拼接标签的数据
    # 特征9 = np.resize(np.array(工作表.range((5, 156), (49, 156)).value), (45, 1))  # 这个是没拼接标签的数据
    # 特征10 = np.resize(np.array(工作表.range((5, 157), (49, 157)).value), (45, 1))  # 这个是没拼接标签的数据
    # 特征11 = np.resize(np.array(工作表.range((5, 162), (49, 162)).value), (45, 1))  # 这个是没拼接标签的数据
    # 特征12 = np.resize(np.array(工作表.range((5, 172), (49, 172)).value), (45, 1))  # 这个是没拼接标签的数据
    # 原始糖肾数据 = np.hstack((特征1, 特征2, 特征3, 特征4, 特征5, 特征6, 特征7, 特征8, 特征9, 特征10, 特征11, 特征12))  # 这个是未拼接标签的数据
    原始糖肾数据 = np.hstack((特征1, 特征2, 特征3, 特征4, 特征5, 特征6))  # 这个是未拼接标签的数据
    标准化糖肾数据 = preprocessing.scale(原始糖肾数据)  # 将数据处理为标准正态分布

    糖肾训练数据, 糖肾测试数据, 糖肾训练标签, 糖肾测试标签 = train_test_split(标准化糖肾数据, OCTA糖肾标签, random_state=1, test_size=5, shuffle=True)
    C值, gamma值 = np.arange(0.5, 10, 0.0625), np.arange(0.5, 10, 0.5)
    超参数, 模型 = {'kernel': ('linear', 'rbf'), 'C': C值, 'gamma': gamma值}, svm.SVC(probability=True)
    参数优化器 = GridSearchCV(模型, 超参数, scoring = 'roc_auc_ovo', n_jobs=-1, cv=5)
    参数优化器.fit(糖肾训练数据, 糖肾训练标签.ravel())
    print("最佳参数", 参数优化器.best_params_)
    print("最好评分", 参数优化器.best_score_)


def 糖肾分类(工作表):
    # 读数据的时候，数据的索引是从矩形区域的左上角到矩形区域的右下角，第一个是左上角坐标，第二个是右下角坐标
    OCTA糖肾标签 = np.array(工作表.range((5, 5), (49, 5)).value)  # 分两类
    OCTA糖肾标签.resize(45, 1)
    类别 = ['无糖肾', '糖肾']
    # 特征1 = np.resize(np.array(工作表.range((5, 56), (49, 56)).value), (45, 1))  # 这个是没拼接标签的数据
    特征2 = np.resize(np.array(工作表.range((5, 57), (49, 57)).value), (45, 1))  # 这个是没拼接标签的数据
    # 特征3 = np.resize(np.array(工作表.range((5, 75), (49, 75)).value), (45, 1))  # 这个是没拼接标签的数据
    # 特征4 = np.resize(np.array(工作表.range((5, 80), (49, 80)).value), (45, 1))  # 这个是没拼接标签的数据
    # 特征5 = np.resize(np.array(工作表.range((5, 88), (49, 88)).value), (45, 1))  # 这个是没拼接标签的数据
    # 特征6 = np.resize(np.array(工作表.range((5, 90), (49, 90)).value), (45, 1))  # 这个是没拼接标签的数据
    # 特征7 = np.resize(np.array(工作表.range((5, 141), (49, 141)).value), (45, 1))  # 这个是没拼接标签的数据
    # 特征8 = np.resize(np.array(工作表.range((5, 147), (49, 147)).value), (45, 1))  # 这个是没拼接标签的数据
    # 特征9 = np.resize(np.array(工作表.range((5, 156), (49, 156)).value), (45, 1))  # 这个是没拼接标签的数据
    # 特征10 = np.resize(np.array(工作表.range((5, 157), (49, 157)).value), (45, 1))  # 这个是没拼接标签的数据
    # 特征11 = np.resize(np.array(工作表.range((5, 162), (49, 162)).value), (45, 1))  # 这个是没拼接标签的数据
    特征12 = np.resize(np.array(工作表.range((5, 172), (49, 172)).value), (45, 1))  # 这个是没拼接标签的数据
    # 原始糖肾数据 = np.hstack((特征1, 特征3, 特征4, 特征6, 特征7, 特征9, 特征10, 特征11))  # 这个是未拼接标签的数据
    原始糖肾数据 = np.hstack((特征2, 特征12))  # 这个是未拼接标签的数据
    标准化糖肾数据 = preprocessing.scale(原始糖肾数据)  # 将数据处理为标准正态分布

    糖肾训练数据, 糖肾测试数据, 糖肾训练标签, 糖肾测试标签 = train_test_split(标准化糖肾数据, OCTA糖肾标签, test_size=5, shuffle=True)
    # print(糖肾训练标签)
    # print(糖肾测试标签)

    '''使用线性核得到特征的重要性'''

    糖肾分类器 = svm.SVC(C=0.5, kernel='linear', gamma=0.5, probability=True,decision_function_shape='ovo')  # ovr：一对多策略，ovo：一对一策略
    糖肾分类器.fit(糖肾训练数据, 糖肾训练标签.ravel())  # ravel函数在降维时默认是行序优先
    # 计算糖肾分类器的准确率
    训练准确率, 测试准确率 = 糖肾分类器.score(糖肾训练数据, 糖肾训练标签), 糖肾分类器.score(糖肾测试数据, 糖肾测试标签)
    训练AUC = roc_auc_score(糖肾训练标签.ravel(), 糖肾分类器.decision_function(糖肾训练数据)) # 二分类用decision_function
    测试AUC = roc_auc_score(糖肾测试标签.ravel(), 糖肾分类器.decision_function(糖肾测试数据))
    print("糖肾测试集AUC：", 测试AUC)
    print("糖肾训练集准确率：", 训练准确率)
    print("糖肾测试集准确率：", 测试准确率)
    # '''
    print("支持向量的数量", 糖肾分类器.n_support_)
    print("支持向量的索引", 糖肾分类器.support_)
    # print("特征重要性：",'\n' , 糖肾分类器.coef_)
    print(metrics.classification_report(糖肾测试标签, 糖肾分类器.predict(糖肾测试数据), target_names=类别))
    metrics.plot_confusion_matrix(糖肾分类器, 糖肾测试数据, 糖肾测试标签)
    plt.show()
    # '''

def 部份特征糖肾分类(工作表):
    # 读数据的时候，数据的索引是从矩形区域的左上角到矩形区域的右下角，第一个是左上角坐标，第二个是右下角坐标
    OCTA糖肾标签 = np.array(工作表.range((5, 5), (49, 5)).value)  # 分两类
    OCTA糖肾标签.resize(45, 1)
    类别 = ['无糖肾', '糖肾']
    特征1 = np.resize(np.array(工作表.range((5, 56), (49, 56)).value), (45, 1))  # 这个是没拼接标签的数据
    特征2 = np.resize(np.array(工作表.range((5, 57), (49, 57)).value), (45, 1))  # 这个是没拼接标签的数据
    特征3 = np.resize(np.array(工作表.range((5, 75), (49, 75)).value), (45, 1))  # 这个是没拼接标签的数据
    特征4 = np.resize(np.array(工作表.range((5, 80), (49, 80)).value), (45, 1))  # 这个是没拼接标签的数据
    特征5 = np.resize(np.array(工作表.range((5, 88), (49, 88)).value), (45, 1))  # 这个是没拼接标签的数据
    特征6 = np.resize(np.array(工作表.range((5, 90), (49, 90)).value), (45, 1))  # 这个是没拼接标签的数据
    特征7 = np.resize(np.array(工作表.range((5, 141), (49, 141)).value), (45, 1))  # 这个是没拼接标签的数据
    特征8 = np.resize(np.array(工作表.range((5, 147), (49, 147)).value), (45, 1))  # 这个是没拼接标签的数据
    特征9 = np.resize(np.array(工作表.range((5, 156), (49, 156)).value), (45, 1))  # 这个是没拼接标签的数据
    特征10 = np.resize(np.array(工作表.range((5, 157), (49, 157)).value), (45, 1))  # 这个是没拼接标签的数据
    特征11 = np.resize(np.array(工作表.range((5, 162), (49, 162)).value), (45, 1))  # 这个是没拼接标签的数据
    特征12 = np.resize(np.array(工作表.range((5, 172), (49, 172)).value), (45, 1))  # 这个是没拼接标签的数据
    原始糖肾数据 = np.hstack((特征1, 特征2, 特征3, 特征4, 特征5, 特征6, 特征7, 特征8, 特征9, 特征10, 特征11, 特征12))  # 这个是未拼接标签的数据
    标准化糖肾数据 = preprocessing.scale(原始糖肾数据)  # 将数据处理为标准正态分布

    糖肾训练数据, 糖肾测试数据, 糖肾训练标签, 糖肾测试标签 = train_test_split(标准化糖肾数据, OCTA糖肾标签, random_state=1, test_size=5, shuffle=True)

    '''使用线性核得到特征的重要性'''

    糖肾分类器 = svm.SVC(C=0.5, kernel='linear', gamma=0.5, probability=True,decision_function_shape='ovo')  # ovr：一对多策略，ovo：一对一策略
    糖肾分类器.fit(糖肾训练数据, 糖肾训练标签.ravel())  # ravel函数在降维时默认是行序优先
    # 计算糖肾分类器的准确率
    训练准确率, 测试准确率 = 糖肾分类器.score(糖肾训练数据, 糖肾训练标签), 糖肾分类器.score(糖肾测试数据, 糖肾测试标签)
    训练AUC = roc_auc_score(糖肾训练标签.ravel(), 糖肾分类器.decision_function(糖肾训练数据)) # 二分类用decision_function
    测试AUC = roc_auc_score(糖肾测试标签.ravel(), 糖肾分类器.decision_function(糖肾测试数据))
    print("糖肾训练集AUC：", 训练AUC)
    print("糖肾测试集AUC：", 测试AUC)
    print("糖肾训练集准确率：", 训练准确率)
    print("糖肾测试集准确率：", 测试准确率)
    # '''
    print("支持向量的数量", 糖肾分类器.n_support_)
    print("支持向量的索引", 糖肾分类器.support_)
    # print("特征重要性：",'\n' , 糖肾分类器.coef_)
    print(metrics.classification_report(糖肾测试标签, 糖肾分类器.predict(糖肾测试数据), target_names=类别))
    metrics.plot_confusion_matrix(糖肾分类器, 糖肾测试数据, 糖肾测试标签)
    plt.show()
    # '''



def 特征选择(工作表):
    # 读数据的时候，数据的索引是从矩形区域的左上角到矩形区域的右下角，第一个是左上角坐标，第二个是右下角坐标
    OCTA糖肾标签 = np.array(工作表.range((5, 5), (49, 5)).value)  # 分两类
    OCTA糖肾标签.resize(45, 1)
    OCTA黄斑检查右 = np.array(工作表.range((5, 40), (49, 50)).value)
    OCTA血流密度右 = np.array(工作表.range((5, 56), (49, 90)).value)

    OCTA黄斑检查左 = np.array(工作表.range((5, 121), (49, 131)).value)  # 推测是左眼
    OCTA血流密度左 = np.array(工作表.range((5, 138), (49, 172)).value)  # 推测是左眼
    原始糖肾数据 = np.hstack((OCTA黄斑检查右, OCTA血流密度右, OCTA黄斑检查左, OCTA血流密度左))  # 这个是没拼接标签的数据
    print(原始糖肾数据.shape)

    标准化糖肾数据 = preprocessing.scale(原始糖肾数据)  # 将数据处理为标准正态分布
    糖肾训练数据, 糖肾测试数据, 糖肾训练标签, 糖肾测试标签 = train_test_split(标准化糖肾数据, OCTA糖肾标签, random_state=1, test_size=5, shuffle=True)

    特征重要性排序 = np.zeros(92)

    for i in np.arange(0.5, 10, 0.5):
        评估器 = svm.SVC(C=i, kernel='linear', decision_function_shape='ovo')
        特征选择器 = RFECV(评估器, min_features_to_select=1, cv=5)
        特征选择器.fit(糖肾训练数据, 糖肾训练标签.ravel())
        for j in range(0, 92):
            特征重要性排序[j] += 特征选择器.ranking_[j]
    for i in 特征重要性排序:
        print(i)
    # '''

def 多因素方差分析显著性检验(工作表):
    # 读数据的时候，数据的索引是从矩形区域的左上角到矩形区域的右下角，第一个是左上角坐标，第二个是右下角坐标
    OCTA糖肾标签 = np.array(工作表.range((5, 5), (49, 5)).value)  # 分两类
    OCTA糖肾标签.resize(45, 1)
    OCTA黄斑检查右 = np.array(工作表.range((5, 40), (49, 50)).value)
    # OCTA血流密度右 = np.array(工作表.range((5, 56), (49, 90)).value)

    OCTA黄斑检查左 = np.array(工作表.range((5, 121), (49, 131)).value)  # 推测是左眼
    # OCTA血流密度左 = np.array(工作表.range((5, 138), (49, 172)).value)  # 推测是左眼
    # 原始糖肾数据 = np.hstack((OCTA黄斑检查右, OCTA血流密度右, OCTA黄斑检查左, OCTA血流密度左))  # 这个是未拼接标签的数据

    特征1 = np.resize(np.array(工作表.range((5, 56), (49, 56)).value), (45, 1))  # 这个是没拼接标签的数据
    特征2 = np.resize(np.array(工作表.range((5, 57), (49, 57)).value), (45, 1))  # 这个是没拼接标签的数据
    特征3 = np.resize(np.array(工作表.range((5, 75), (49, 75)).value), (45, 1))  # 这个是没拼接标签的数据
    特征4 = np.resize(np.array(工作表.range((5, 80), (49, 80)).value), (45, 1))  # 这个是没拼接标签的数据
    特征5 = np.resize(np.array(工作表.range((5, 88), (49, 88)).value), (45, 1))  # 这个是没拼接标签的数据
    特征6 = np.resize(np.array(工作表.range((5, 90), (49, 90)).value), (45, 1))  # 这个是没拼接标签的数据
    特征7 = np.resize(np.array(工作表.range((5, 141), (49, 141)).value), (45, 1))  # 这个是没拼接标签的数据
    特征8 = np.resize(np.array(工作表.range((5, 147), (49, 147)).value), (45, 1))  # 这个是没拼接标签的数据
    特征9 = np.resize(np.array(工作表.range((5, 156), (49, 156)).value), (45, 1))  # 这个是没拼接标签的数据
    特征10 = np.resize(np.array(工作表.range((5, 157), (49, 157)).value), (45, 1))  # 这个是没拼接标签的数据
    特征11 = np.resize(np.array(工作表.range((5, 162), (49, 162)).value), (45, 1))  # 这个是没拼接标签的数据
    特征12 = np.resize(np.array(工作表.range((5, 172), (49, 172)).value), (45, 1))  # 这个是没拼接标签的数据

    原始糖肾数据 = np.hstack((特征1, 特征2, 特征3, 特征4, 特征5, 特征6, 特征7, 特征8, 特征9, 特征10, 特征11, 特征12))  # 这个是未拼接标签的数据

    拼接标签的原始糖肾数据 = np.hstack((原始糖肾数据, OCTA糖肾标签))

    pd形式数据 = pd.DataFrame(拼接标签的原始糖肾数据,
                          columns=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
                                   'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20',
                                   'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30',
                                   'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37', 'A38', 'A39', 'A40',
                                   'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49', 'A50',
                                   'A51', 'A52', 'A53', 'A54', 'A55', 'A56', 'A57', 'A58', 'A59', 'A60',
                                   'A61', 'A62', 'A63', 'A64', 'A65', 'A66', 'A67', 'A68', 'A69', 'A70',
                                   'A71', 'A72', 'A73', 'A74', 'A75', 'A76', 'A77', 'A78', 'A79', 'A80',
                                   'A81', 'A82', 'A83', 'A84', 'A85', 'A86', 'A87', 'A88', 'A89', 'A90',
                                   'A91', 'A92',  'label'])
    # print('\n', pd形式数据)

    # 多因素方差分析，不考虑各个因素之间的相互作用
    # '''
    # 不考虑因素之间的相互作用 = 'label ~ A1+A2+A3+A4+A5+A6+A7+A8+A9+A10+A11+A12+A13+A14+A15+A16+A17+A18+A19+A20+A21+A22+A23+A24+A25+A26+A27+A28+A29+A30+A31+A32+A33+A34+A35+A36+A37+A38+A39+A40+A41+A42+A43+A44'

    不考虑因素之间的相互作用 = 'label ~ A50+A51+A52+A53+A54+A55+A56+A57+A58+A59+A60+A61+A62+A63+A64+A65+A66+A67+A68+A69+A70'


    model = ols(不考虑因素之间的相互作用, pd形式数据).fit()
    anovat = anova_lm(model)
    print("不考虑因素之间的影响",'\n' , anovat)
    # '''

def 部份特征多因素方差分析显著性检验(工作表):
    # 读数据的时候，数据的索引是从矩形区域的左上角到矩形区域的右下角，第一个是左上角坐标，第二个是右下角坐标
    OCTA糖肾标签 = np.array(工作表.range((5, 5), (49, 5)).value)  # 分两类
    OCTA糖肾标签.resize(45, 1)

    特征1 = np.resize(np.array(工作表.range((5, 56), (49, 56)).value), (45, 1))  # 这个是没拼接标签的数据
    特征2 = np.resize(np.array(工作表.range((5, 57), (49, 57)).value), (45, 1))  # 这个是没拼接标签的数据
    特征3 = np.resize(np.array(工作表.range((5, 75), (49, 75)).value), (45, 1))  # 这个是没拼接标签的数据
    特征4 = np.resize(np.array(工作表.range((5, 80), (49, 80)).value), (45, 1))  # 这个是没拼接标签的数据
    特征5 = np.resize(np.array(工作表.range((5, 88), (49, 88)).value), (45, 1))  # 这个是没拼接标签的数据
    特征6 = np.resize(np.array(工作表.range((5, 90), (49, 90)).value), (45, 1))  # 这个是没拼接标签的数据
    特征7 = np.resize(np.array(工作表.range((5, 141), (49, 141)).value), (45, 1))  # 这个是没拼接标签的数据
    特征8 = np.resize(np.array(工作表.range((5, 147), (49, 147)).value), (45, 1))  # 这个是没拼接标签的数据
    特征9 = np.resize(np.array(工作表.range((5, 156), (49, 156)).value), (45, 1))  # 这个是没拼接标签的数据
    特征10 = np.resize(np.array(工作表.range((5, 157), (49, 157)).value), (45, 1))  # 这个是没拼接标签的数据
    特征11 = np.resize(np.array(工作表.range((5, 162), (49, 162)).value), (45, 1))  # 这个是没拼接标签的数据
    特征12 = np.resize(np.array(工作表.range((5, 172), (49, 172)).value), (45, 1))  # 这个是没拼接标签的数据

    原始糖肾数据 = np.hstack((特征1, 特征2, 特征3, 特征4, 特征5, 特征6, 特征7, 特征8, 特征9, 特征10, 特征11, 特征12))  # 这个是未拼接标签的数据

    拼接标签的原始糖肾数据 = np.hstack((原始糖肾数据, OCTA糖肾标签))

    pd形式数据 = pd.DataFrame(拼接标签的原始糖肾数据,
                          columns=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10','A11', 'A12', 'label'])
    # print('\n', pd形式数据)

    # 多因素方差分析，不考虑各个因素之间的相互作用
    # '''
    # 不考虑因素之间的相互作用 = 'label ~ A1+A2+A3+A4+A5+A6+A7+A8+A9+A10+A11+A12+A13+A14+A15+A16+A17+A18+A19+A20+A21+A22+A23+A24+A25+A26+A27+A28+A29+A30+A31+A32+A33+A34+A35+A36+A37+A38+A39+A40+A41+A42+A43+A44'

    不考虑因素之间的相互作用 = 'label ~ A1+A2+A3+A4+A5+A6+A7+A8+A9+A10+A11+A12'


    model = ols(不考虑因素之间的相互作用, pd形式数据).fit()
    anovat = anova_lm(model)
    print("不考虑因素之间的影响",'\n' , anovat)
    # '''


if __name__ == '__main__':
    # 先用HBA1C（糖化血红蛋白）到血红蛋白HGB
    肾病病例工作表 = xw.Book('友谊肾穿入组78例临床数据202125给李煜老师副本.xls').sheets(1)
    # 参数搜索(肾病病例工作表)
    糖肾分类(肾病病例工作表)
    # 特征选择(肾病病例工作表)
    # 多因素方差分析显著性检验(肾病病例工作表)
    # 部份特征多因素方差分析显著性检验(肾病病例工作表)