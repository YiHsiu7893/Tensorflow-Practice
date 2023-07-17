import tensorflow as tf              
import numpy as np                    

# 建立訓練資料
x1 = np.random.random((500, 1))        # 產生500個在0~1之間的亂數
x2 = np.random.random((500, 1))+1      # 產生500個在1~2之間的亂數
x_train = np.concatenate((x1, x2))     # 產生訓練1000個資料的因

y1 = np.zeros((500,), dtype=int)       # 產生500個0
y2 = np.ones((500,), dtype=int)        # 產生500個1
y_train = np.concatenate((y1, y2))     # 產生訓練1000個資料的結果

# 建立MLP模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=10, activation=tf.nn.relu, input_dim=1),
    tf.keras.layers.Dense(units=10, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=2, activation=tf.nn.softmax)
])
# 另一種建立模型方式
#model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.relu, input_dim=1))
#model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.relu))
#model.add(tf.keras.layers.Dense(units=2, activation=tf.nn.softmax))

# 編譯模型
model.compile(optimizer='adam',                        # 編譯使用adam最佳化
              loss='sparse_categorical_crossentropy',  # 損失率使用稀疏分類別交叉熵
              metrics=['accuracy'])                    # 設定編譯處理時以正確率為主

# 訓練模型
model.fit(x=x_train, y=y_train,        # 設定訓練的因和果的資料
          epochs=20,                   # 設定訓練的次數，就是機器學習的次數
          batch_size=128)              # 設定每次訓練的筆數

# 建立測試資料&評估正確率
x_test = np.array([[0.22], [0.31], [1.22], [1.33]])    # 自訂測試資料的因
y_test = np.array([0, 0, 1, 1])                        # 自訂測試資料的結果
score = model.evaluate(x_test, y_test, batch_size=128) # 計算測試正確率
print("score:", score)

# 預測
predict = model.predict(x_test)        # 取得每一個預測結果的機率
print("predict:", predict)
print("Ans:", np.argmax(predict[0]), np.argmax(predict[1]), np.argmax(predict[2]), np.argmax(predict[3]))
# 輸出每一個預測結果的答案
print("y_test:", y_test)               # 實際的測試答案