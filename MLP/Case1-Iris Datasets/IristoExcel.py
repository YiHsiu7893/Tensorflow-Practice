import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import xlsxwriter
import pandas as pd

# 取得鳶尾花的資料
iris = datasets.load_iris()
print("iris.data.shape=", iris.data.shape)  # (150, 4)
print("dir(iris):", dir(iris))
# ['DESCR', 'data', 'data_module', 'feature_names', 'filename', 'frame', 'target', 'target_names']
print("iris.target.shape=", iris.target.shape) # (150,)
try:
    print("iris.feature_names=", iris.feature_names)  # 查看特徵值名稱
except:
    print("No iris.features_names")

# 轉換資料型態
try:
    df = pd.DataFrame(iris.data, columns=iris.feature_names)  # 處理特徵值
except:
    df = pd.DataFrame(iris.data, columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
df['target'] = iris.target  # 處理結果Target

# 儲存到csv
df.to_csv("iris.csv", sep='\t')
# 儲存到Excel
writer = pd.ExcelWriter('iris.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1')
writer.close()