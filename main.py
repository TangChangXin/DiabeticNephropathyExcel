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


def 参数搜索(工作表):
    # 读数据的时候，数据的索引是从矩形区域的左上角到矩形区域的右下角，第一个是左上角坐标，第二个是右下角坐标
    糖肾标签 = np.array(工作表.range((2, 6), (152, 6)).value)  # 分三类：无糖肾、糖肾、二者合并。
    糖肾标签.resize(151, 1)
    原始糖肾数据 = np.array(工作表.range((2, 30), (152, 50)).value)  # 这个是没拼接标签的数据
    标准化糖肾数据 = preprocessing.scale(原始糖肾数据)  # 将数据处理为标准正态分布

    # 共152人，0.9:0.1   136人训练，16人
    糖肾训练数据, 糖肾测试数据, 糖肾训练标签, 糖肾测试标签 = train_test_split(标准化糖肾数据, 糖肾标签, random_state=1, test_size=16, shuffle=True)
    C值, gamma值 = np.arange(0.5, 10, 0.0625), np.arange(0.5, 10, 0.5)
    超参数, 模型 = {'kernel': ('linear', 'rbf'), 'C': C值, 'gamma': gamma值}, svm.SVC(probability=True) #
    # 超参数, 模型 = {'kernel': ['linear'], 'C': C值}, svm.SVC(probability=True)  # 只用线性核搜索
    参数优化器 = GridSearchCV(模型, 超参数, scoring = 'roc_auc_ovo', n_jobs=-1, cv=5)
    参数优化器.fit(糖肾训练数据, 糖肾训练标签.ravel())
    print("最佳参数", 参数优化器.best_params_)
    print("最好评分", 参数优化器.best_score_)

def 部份参数搜索(工作表):
    # 读数据的时候，数据的索引是从矩形区域的左上角到矩形区域的右下角，第一个是左上角坐标，第二个是右下角坐标
    糖肾标签 = np.array(工作表.range((2, 6), (152, 6)).value)  # 分三类：无糖肾、糖肾、二者合并。
    糖肾标签.resize(151, 1)
    特征1 = np.resize(np.array(工作表.range((2, 30), (152, 30)).value), (151, 1))  # 这个是没拼接标签的数据
    特征2 = np.resize(np.array(工作表.range((2, 38), (152, 38)).value), (151, 1))  # 这个是没拼接标签的数据
    特征3 = np.resize(np.array(工作表.range((2, 43), (152, 43)).value), (151, 1))  # 这个是没拼接标签的数据
    特征4 = np.resize(np.array(工作表.range((2, 45), (152, 45)).value), (151, 1))  # 这个是没拼接标签的数据
    特征5 = np.resize(np.array(工作表.range((2, 41), (152, 41)).value), (151, 1))  # 这个是没拼接标签的数据
    特征6 = np.resize(np.array(工作表.range((2, 44), (152, 44)).value), (151, 1))  # 这个是没拼接标签的数据
    原始糖肾数据 = np.hstack((特征1, 特征2, 特征3, 特征4, 特征5, 特征6)) # 这个是没拼接标签的数据
    标准化糖肾数据 = preprocessing.scale(原始糖肾数据)  # 将数据处理为标准正态分布

    # 共152人，0.9:0.1   136人训练，16人
    糖肾训练数据, 糖肾测试数据, 糖肾训练标签, 糖肾测试标签 = train_test_split(标准化糖肾数据, 糖肾标签, random_state=1, test_size=16, shuffle=True)
    C值, gamma值 = np.arange(0.5, 10, 0.0625), np.arange(0.5, 10, 0.5)
    超参数, 模型 = {'kernel': ('linear', 'rbf'), 'C': C值, 'gamma': gamma值}, svm.SVC(probability=True) #
    # 超参数, 模型 = {'kernel': ['linear'], 'C': C值}, svm.SVC(probability=True)  # 只用线性核搜索
    参数优化器 = GridSearchCV(模型, 超参数, scoring = 'roc_auc_ovo', n_jobs=-1, cv=5)
    参数优化器.fit(糖肾训练数据, 糖肾训练标签.ravel())
    print("最佳参数", 参数优化器.best_params_)
    print("最好评分", 参数优化器.best_score_)


def 糖肾分类(工作表):
    # 读数据的时候，数据的索引是从矩形区域的左上角到矩形区域的右下角，第一个是左上角坐标，第二个是右下角坐标
    糖肾标签 = np.array(工作表.range((2, 6), (152, 6)).value)  # 分三类：无糖肾、糖肾、二者合并。
    糖肾标签.resize(151, 1)
    类别 = ['无糖肾', '糖肾', '二者合并']
    原始糖肾数据 = np.array(工作表.range((2, 30), (152, 50)).value)  # 这个是没拼接标签的数据
    标准化糖肾数据 = preprocessing.scale(原始糖肾数据)  # 将数据处理为标准正态分布

    # 共152人，0.9:0.1   136人训练，16人
    糖肾训练数据, 糖肾测试数据, 糖肾训练标签, 糖肾测试标签 = train_test_split(标准化糖肾数据, 糖肾标签, test_size=16, shuffle=True)

    '''使用线性核得到特征的重要性'''

    # 糖肾分类器 = svm.SVC(C=0.8125, kernel='linear', gamma=0.5, probability=True,decision_function_shape='ovo')  # ovr：一对多策略，ovo：一对一策略
    糖肾分类器 = svm.SVC(C=0.5, kernel='rbf', gamma=0.5, probability=True,
                    decision_function_shape='ovo')  # ovr：一对多策略，ovo：一对一策略
    糖肾分类器.fit(糖肾训练数据, 糖肾训练标签.ravel())  # ravel函数在降维时默认是行序优先
    # 计算糖肾分类器的准确率
    训练准确率, 测试准确率 = 糖肾分类器.score(糖肾训练数据, 糖肾训练标签), 糖肾分类器.score(糖肾测试数据, 糖肾测试标签)
    训练AUC = roc_auc_score(糖肾训练标签.ravel(), 糖肾分类器.predict_proba(糖肾训练数据), multi_class='ovo')
    测试AUC = roc_auc_score(糖肾测试标签.ravel(), 糖肾分类器.predict_proba(糖肾测试数据), multi_class='ovo')
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


# 选择部分特征来分类
def 部份特征糖肾分类(工作表):
    # 读数据的时候，数据的索引是从矩形区域的左上角到矩形区域的右下角，第一个是左上角坐标，第二个是右下角坐标
    糖肾标签 = np.array(工作表.range((2, 6), (152, 6)).value)  # 分三类：无糖肾、糖肾、二者合并。
    糖肾标签.resize(151, 1)
    类别 = ['无糖肾', '糖肾', '二者合并']
    特征1 = np.resize(np.array(工作表.range((2, 30), (152, 30)).value), (151, 1))  # 这个是没拼接标签的数据
    特征2 = np.resize(np.array(工作表.range((2, 38), (152, 38)).value), (151, 1))  # 这个是没拼接标签的数据
    特征3 = np.resize(np.array(工作表.range((2, 43), (152, 43)).value), (151, 1))  # 这个是没拼接标签的数据
    特征4 = np.resize(np.array(工作表.range((2, 45), (152, 45)).value), (151, 1))  # 这个是没拼接标签的数据
    特征5 = np.resize(np.array(工作表.range((2, 41), (152, 41)).value), (151, 1))  # 这个是没拼接标签的数据
    特征6 = np.resize(np.array(工作表.range((2, 44), (152, 44)).value), (151, 1))  # 这个是没拼接标签的数据
    原始糖肾数据 = np.hstack((特征1, 特征2, 特征3, 特征4, 特征5, 特征6))
    print(原始糖肾数据.shape)
    标准化糖肾数据 = preprocessing.scale(原始糖肾数据)  # 将数据处理为标准正态分布

    # 共152人，0.9:0.1   136人训练，16人
    糖肾训练数据, 糖肾测试数据, 糖肾训练标签, 糖肾测试标签 = train_test_split(标准化糖肾数据, 糖肾标签, test_size=16, shuffle=True)

    '''使用线性核得到特征的重要性'''

    糖肾分类器 = svm.SVC(C=6.875, kernel='linear', gamma=9.5, probability=True, decision_function_shape='ovo')  # ovr：一对多策略，ovo：一对一策略
    # 糖肾分类器 = svm.SVC(C=0.5, kernel='rbf', gamma=0.5, probability=True, decision_function_shape='ovo')  # ovr：一对多策略，ovo：一对一策略
    糖肾分类器.fit(糖肾训练数据, 糖肾训练标签.ravel())  # ravel函数在降维时默认是行序优先
    # 计算糖肾分类器的准确率
    训练准确率, 测试准确率 = 糖肾分类器.score(糖肾训练数据, 糖肾训练标签), 糖肾分类器.score(糖肾测试数据, 糖肾测试标签)
    训练AUC = roc_auc_score(糖肾训练标签.ravel(), 糖肾分类器.predict_proba(糖肾训练数据), multi_class='ovo')
    测试AUC = roc_auc_score(糖肾测试标签.ravel(), 糖肾分类器.predict_proba(糖肾测试数据), multi_class='ovo')
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
    糖肾标签 = np.array(工作表.range((2, 6), (152, 6)).value)  # 分三类：无糖肾、糖肾、二者合并。
    糖肾标签.resize(151, 1)
    原始糖肾数据 = np.array(工作表.range((2, 30), (152, 50)).value)  # 这个是没拼接标签的数据
    标准化糖肾数据 = preprocessing.scale(原始糖肾数据)  # 将数据处理为标准正态分布
    特征重要性排序 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # 共152人，0.9:0.1   136人训练，16人
    糖肾训练数据, 糖肾测试数据, 糖肾训练标签, 糖肾测试标签 = train_test_split(标准化糖肾数据, 糖肾标签, random_state=1, test_size=16, shuffle=True)
    for i in np.arange(0.5, 10, 0.5):
        评估器 = svm.SVC(C=i, kernel='linear', decision_function_shape='ovo')
        特征选择器 = RFECV(评估器, min_features_to_select=1, cv=5)
        特征选择器.fit(糖肾训练数据, 糖肾训练标签.ravel())
        for j in range(0, 21):
            特征重要性排序[j] += 特征选择器.ranking_[j]
    print(特征重要性排序)


def 最优参数时特征选择(工作表):
    # 读数据的时候，数据的索引是从矩形区域的左上角到矩形区域的右下角，第一个是左上角坐标，第二个是右下角坐标
    糖肾标签 = np.array(工作表.range((2, 6), (152, 6)).value)  # 分三类：无糖肾、糖肾、二者合并。
    糖肾标签.resize(151, 1)
    原始糖肾数据 = np.array(工作表.range((2, 30), (152, 50)).value)  # 这个是没拼接标签的数据
    标准化糖肾数据 = preprocessing.scale(原始糖肾数据)  # 将数据处理为标准正态分布

    # 共152人，0.9:0.1   136人训练，16人
    糖肾训练数据, 糖肾测试数据, 糖肾训练标签, 糖肾测试标签 = train_test_split(标准化糖肾数据, 糖肾标签, random_state=1, test_size=16, shuffle=True)
    评估器 = svm.SVC(C=0.8125, kernel='linear', decision_function_shape='ovo')
    特征选择器 = RFECV(评估器, min_features_to_select=1, cv=5)
    特征选择器.fit(糖肾训练数据, 糖肾训练标签.ravel())
    print(type(特征选择器.ranking_))

if __name__ == '__main__':
    # 先用HBA1C（糖化血红蛋白）到血红蛋白HGB
    肾病病例工作表 = xw.Book('肾穿病例整理原始数据_删除了一些有空值的数据_副本.xlsx').sheets(1)
    # 参数搜索(肾病病例工作表)
    # 部份参数搜索(肾病病例工作表)
    # 特征选择(肾病病例工作表)
    # 最优参数时特征选择(肾病病例工作表)
    糖肾分类(肾病病例工作表)
    # 部份特征糖肾分类(肾病病例工作表)

