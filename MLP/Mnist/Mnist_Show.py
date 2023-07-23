"""
brief intro.
查看Mnist手寫數字資料集的資料型態
共三種方式:
1.單張圖片每一點的色彩資料
2.單張圖片的灰階顯示
3.一次查看多張圖片
"""
import tensorflow as tf
import matplotlib.pyplot as plt

# 函式printMatrix()，查看圖片檔的每一點色彩資料
def printMatrix(a):
    rows = a.shape[0]                           # 取得圖片高度
    cols = a.shape[1]                           # 取得圖片寬度
    for i in range (0, rows):                   # 處理每一列的每一點資料
        str1 = ""
        for j in range (0, cols):               # 處理同一列的每一點資料
            str1 = str1 + ("%3.0f " % a[i, j])  # 取得單點資料
        print(str1)                             # 輸出整列的資料
    print("")

# 取得手寫數字資料
(x_train, y_train), (x_test, y_test) =tf.keras.datasets.mnist.load_data()

# 查看第num筆的圖片
num = 0  
# 圖片色彩資料及標籤
printMatrix(x_train[num])
print("y_train[%d] = " % (num), y_train[num])
# 以灰階顯示圖片
plt.title("x_train[%d] Label: %d" % (num, y_train[num]))
plt.imshow(x_train[num], cmap=plt.get_cmap('gray_r'))
plt.show()

# 查看多筆(0~15)圖片
for num in range(0, 16):
    plt.subplot(4, 4, num+1)
    plt.title("[%d] Label: %d" % (num, y_train[num]))
    plt.imshow(x_train[num], cmap=plt.get_cmap('gray_r'))
plt.show()