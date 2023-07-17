"""
brief intro.
與MLP_Start.py差不多，但此範例將標籤做了單熱編碼預處理
One-Hot Encoding係指將標籤轉換為二進制變量，優點是增進模型處理效率
*使用One-Hot Encoding亦須修改loss參數
"""
import tensorflow as tf
import numpy as np

# 建立訓練資料
x1 = np.random.random((500, 1))
x2 = np.random.random((500, 1))+1
x_train = np.concatenate((x1, x2))
y1 = np.zeros((500,), dtype=int)
y2 = np.ones((500,), dtype=int)
y_train = np.concatenate((y1, y2))
# 將訓練結果轉為One-Hot Encoding單熱編碼
y_train2 = tf.keras.utils.to_categorical(y_train, num_classes=2)

# 建立模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=10, activation=tf.nn.relu, input_dim=1),
    tf.keras.layers.Dense(units=10, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=2, activation=tf.nn.softmax)
])

# 編譯模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy,  # loss使用多分類別損失函式
              metrics=['accuracy'])

# 訓練模型
model.fit(x=x_train, y=y_train2,
          epochs=20,
          batch_size=128)

# 建立測試資料
x_test = np.array([[0.22], [0.31], [1.22], [1.33]])
y_test = np.array([0, 0, 1, 1])
# 將測試結果轉為One-Hot Encoding單熱編碼
y_test2 = tf.keras.utils.to_categorical(y_test, num_classes=2)

# 評估正確率
score = model.evaluate(x_test, y_test2, batch_size=128)
print("score:", score)

# 預測
predict = model.predict(x_test)
print("predict:", predict)
print("Ans:", np.argmax(predict[0]), np.argmax(predict[1]), np.argmax(predict[2]), np.argmax(predict[3]))
print("y_test:", y_test)
