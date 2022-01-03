import xlwings as xw
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import matplotlib


# 先用32列到50列的数据试试   HBA1C（糖化血红蛋白）到血红蛋白HGB
肾病病例工作表 = xw.Book('肾穿病例整理原始数据_删除了一些有空值的数据_副本.xlsx').sheets(1)

# 读数据的时候，数据的索引是从矩形区域的左上角到矩形区域的右下角，第一个是左上角坐标，第二个是右下角坐标
糖肾标签 = np.array(肾病病例工作表.range((2, 6), (152, 6)).value)  # 分三类：无糖肾、糖肾、二者合并。
原始糖肾数据 = np.array(肾病病例工作表.range((2, 36), (152, 50)).value)
标准化糖肾数据 = preprocessing.scale(原始糖肾数据) # 将数据处理为标准正态分布

# 共152人，0.9:0.1   136人训练，16人
糖肾训练数据, 糖肾测试数据, 糖肾训练标签, 糖肾测试标签 = train_test_split(标准化糖肾数据, 糖肾标签, random_state=1, test_size=16, shuffle=True)

糖肾多层感知机分类 = MLPClassifier(
    # 不算输入和输出共有3层
    hidden_layer_sizes=(50, 150,10), activation='relu', solver='lbfgs', alpha=1e-05, batch_size='auto',
    learning_rate='constant', max_iter=200, tol=0.0001, verbose=False, warm_start=False, validation_fraction=0.1,
)
糖肾多层感知机分类.fit(糖肾训练数据, 糖肾训练标签)

print('准确率： %s' % 糖肾多层感知机分类.score(糖肾训练数据, 糖肾训练标签))
print('准确率： %s' % 糖肾多层感知机分类.score(糖肾测试数据, 糖肾测试标签))