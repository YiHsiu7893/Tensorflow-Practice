import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 模擬測試資料
x1 = np.random.random((100, 2))*(-4)-1  # 產生100個在-5~-1之間的2D亂數
x2 = np.random.random((100, 2))*4+1     # 產生100個在1~5之間的2D亂數
x_train = np.concatenate((x1, x2))

y1 = np.zeros((100,), dtype=int)        # 產生100個0
y2 = np.ones((100,), dtype=int)         # 產生100個1
y_train = np.concatenate((y1, y2))

# 繪圖-測試資料點
plt.plot(x1[:, 0], x1[:, 1], 'yo')      # 繪出黃點為標籤0
plt.plot(x2[:, 0], x2[:, 1], 'bo')      # 繪出藍點為標籤1
#plt.show()

# 模擬建立模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=1, activation='tanh', input_dim=2))  # 1個神經點，使用tanh
model.add(tf.keras.layers.Dense(units=2, activation='softmax'))

# 模擬訓練模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=x_train, y=y_train,
          epochs=100,
          batch_size=32)
#model.summary()                        # 列印模型

# 觀察權重和偏移量
weights, biases = model.layers[0].get_weights()  # 第一個隱藏層的權重和偏移量
print(weights)
print(biases)

# 繪圖-迴歸線
x = np.linspace(-5, 5, 100)                      # 產生100個在-5到5之間的連續區間數字
y = (-weights[0]*x-biases[0])/weights[1]         # 計算相對應y的位置，ax+by+c=0
plt.axis([-5, 5, -5, 5])                         # 視窗位置
plt.plot(x, y, '-r', label='No.1')               # 文字No.1
plt.title('Graph of y=(%sx+%s)/%s' % (-weights[0], -biases[0], weights[1]))
plt.xlabel('x', color='#1c2833')                 # 列印x
plt.ylabel('y', color='#1c2833')                 # 列印y
plt.legend(loc='upper left')                     # 列印upper left
plt.grid()                                       # 網格
plt.show()