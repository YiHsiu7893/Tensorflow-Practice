import numpy as np
import matplotlib.pyplot as plt

# 圈圈圖
circle1 = np.array([[1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]])
plt.subplot(1, 2, 1)                # 指定繪製在左邊
plt.imshow(circle1, cmap='binary')  # 繪圖

# 叉叉圖
cross1 = np.array([[1, 0, 1],
                   [0, 1, 0],
                   [1, 0, 1]])
plt.subplot(1, 2, 2)                # 指定繪製在右邊
plt.imshow(cross1, cmap='binary')   # 繪圖

# 顯示圈圈和叉叉圖
plt.show()