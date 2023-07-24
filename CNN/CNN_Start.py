"""
brief intro.
建立第一個CNN模型，使用之前做圈叉圖形辨識(MLP)的範例
CNN模型 = Conv2D+MLP，與MLP差異為:
1.加入Conv2D層，套用濾鏡產生更多訓練資料
2.加入Flatten層，將資料轉為1D，再放入MLP
3.特徵值須包含四個維度:(數量, 寬, 高, 顏色)
"""
import tensorflow as tf
import numpy as np

# 建立X, Y資料
circle = np.array([[1, 1, 1],   # 圈圈圖
                   [1, 0, 1],
                   [1, 1, 1]])
cross = np.array([[1, 0, 1],    # 叉叉圖
                  [0, 1, 0],
                  [1, 0, 1]])
X = np.array([circle, cross])   # 定義X
Y = np.array([0, 1])            # 定義Y
category = 2

# Feature, Label預處理
#print(X.shape)  # (2, 3, 3)
X2 = X.reshape(2, 3, 3, 1)      # 改變維度:(數量, 寬, 高, 顏色)
Y2 = tf.keras.utils.to_categorical(Y, num_classes=category)

# 建立CNN模型
model = tf.keras.models.Sequential()
# 加入Conv2D層
model.add(tf.keras.layers.Conv2D(filters=3,               # 一張變三張圖片
                                 kernel_size=(3, 3),      # 濾鏡大小3*3
                                 padding='same',          # 相同大小
                                 activation='relu',       # 激勵函式
                                 input_shape=(3, 3, 1)))  # 輸入每筆數據大小
# 加入Flatten層
model.add(tf.keras.layers.Flatten())                      # 數據轉為1D
model.add(tf.keras.layers.Dense(units=10, activation='relu'))
model.add(tf.keras.layers.Dense(units=category, activation='softmax'))
#model.summary()

# 訓練模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
model.fit(X2, Y2,
          epochs=100)

# 評估正確率
score = model.evaluate(X2, Y2)
print("score:", score)

# 預測
predict = model.predict(X2)
y_predict = np.argmax(predict, axis=1)
print("Ans:", y_predict)
print("  Y:", Y)
