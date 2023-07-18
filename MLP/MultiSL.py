import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 模擬測試資料
x1 = np.random.random((100, 2))*(-4)-1
x2 = np.random.random((100, 2))*4+1
x_train = np.concatenate((x1, x2))
y1 = np.zeros((100,), dtype=int)
y2 = np.ones((100,), dtype=int)
y_train = np.concatenate((y1, y2))

# 繪圖-測試資料點
plt.plot(x_train[:100, 0], x_train[:100, 1], 'yo')  # 繪出黃點為標籤0
plt.plot(x_train[100:, 0], x_train[100:, 1], 'bo')  # 繪出藍點為標籤1

# 模擬建立模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=4, activation='relu', input_dim=2),  # 4個神經點
    tf.keras.layers.Dense(units=2, activation='softmax')
])

# 模擬訓練模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=x_train, y=y_train,
          epochs=50,
          batch_size=32)

# 觀察權重和偏移量
weights, biases = model.layers[0].get_weights()  # 第一個隱藏層的權重和偏移量
print(weights)
print(biases)

# 繪圖-迴歸線
x = np.linspace(-5, 5, 100)
plt.axis([-5, 5, -5, 5])
plt.plot(x, (-weights[0][0]*x-biases[0])/weights[1][0], '-r', label='No.1')  # 第一個神經元
plt.plot(x, (-weights[0][1]*x-biases[1])/weights[1][1], '-g', label='No.2')  # 第二個神經元
plt.plot(x, (-weights[0][2]*x-biases[2])/weights[1][2], '-b', label='No.3')  # 第三個神經元
plt.plot(x, (-weights[0][3]*x-biases[3])/weights[1][3], '-y', label='No.4')  # 第四個神經元
plt.title('Graph of y=(-ax-c)/b')
plt.xlabel('x', color='#1c2833')
plt.ylabel('y', color='#1c2833')
plt.legend(loc='upper left')
plt.grid()
plt.show()