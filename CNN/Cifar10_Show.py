"""
brief intro.
查看Cifar10彩色圖片資料集的資料型態
顯示前36筆
"""
import tensorflow as tf
import matplotlib.pyplot as plt

# 載入彩色圖片資料
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print("x_train = " + str(x_train.shape))  # x_train = (50000, 32, 32, 3)
print("y_train = " + str(y_train.shape))  # y_train = (50000, 1)
print("x_test = " + str(x_test.shape))    # x_test = (10000, 32, 32, 3)
print("y_test = " + str(y_test.shape))    # y_test = (10000, 1)

# 顯示前36筆圖片和標籤
for num in range(0, 36):
    plt.subplot(6, 6, num+1)
    plt.title("[%d]->%d" % (num, y_train[num]))
    plt.imshow(x_train[num])
plt.show()