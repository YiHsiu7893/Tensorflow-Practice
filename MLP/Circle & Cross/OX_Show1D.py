import numpy as np
import matplotlib.pyplot as plt

# 2D圈圈圖
circle1 = np.array([[1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]])
plt.subplot(2, 2, 1)                # 指定繪製在左上
plt.imshow(circle1, cmap='binary')  # 繪圖

# 1D圈圈圖
circle2 = circle1.flatten()         # 2D轉1D方法一
#circle2 = circle1.reshape([9])     # 2D轉1D方法二
print(circle2)
plt.subplot(2, 2, 2)                # 指定繪製在右上
plt.plot(circle2, 'ob')             # 繪圖

# 2D叉叉圖
cross1 = np.array([[1, 0, 1],
                   [0, 1, 0],
                   [1, 0, 1]])
plt.subplot(2, 2, 3)                # 指定繪製在左下
plt.imshow(cross1, cmap='binary')   # 繪圖

# 1D叉叉圖
#cross2 = cross1.flatten()
cross2 = cross1.reshape([9])
print(cross2)
plt.subplot(2, 2, 4)                # 指定繪製在右下
plt.plot(cross2, 'xb')              # 繪圖

# 顯示所有圖形
plt.show()