import tensorflow as tf
import numpy as np

# 取得訓練資料-圈圈圖和叉叉圖
circle1 = np.array([[1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]])
circle2 = circle1.flatten()

cross1 = np.array([[1, 0, 1],
                   [0, 1, 0],
                   [1, 0, 1]])
cross2 = cross1.reshape([9])

X = np.array([circle2, cross2])
Y = np.array([0, 1])
category = 2
dim = 9

# label預處理
Y2 = tf.keras.utils.to_categorical(Y, num_classes=category)

# 建立模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=10, activation='relu', input_dim=dim))
model.add(tf.keras.layers.Dense(units=10, activation='relu'))
model.add(tf.keras.layers.Dense(units=category, activation='softmax'))

# 訓練模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
model.fit(X, Y2,
          epochs=100)

# 評估正確率
score = model.evaluate(X, Y2)
print("score:", score)

# 預測
predict = model.predict(X)
print("predict:", predict)      # 顯示預測數據
y_predict = np.argmax(predict, axis=1)
print("y_predict:", y_predict)  # 顯示預測答案
print("Ans:", Y)                # 顯示實際答案