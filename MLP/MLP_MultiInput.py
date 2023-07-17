import tensorflow as tf
import numpy as np

# 函式CreateDatasets，建立多筆資料
def CreateDatasets(high, iNum, iArraySize):
    x_train = np.random.random((iNum, iArraySize))*float(high)
    # 訓練資料的特徵
    y_train = ((x_train[:iNum, 0]+x_train[:iNum, 1])/2).astype(int)
    # 取整數為訓練資料的標籤
    return x_train, y_train, tf.keras.utils.to_categorical(y_train, num_classes=high)

# 建立訓練資料
category = 10    # 幾種答案
dim = 2          # 每筆資料有2個特徵值X因
x_train, y_train, y_train2 = CreateDatasets(category, 1000, dim)
# 建立全部1000個二維的訓練特徵值X因，而訓練結果Y有10種答案

# 建立模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.relu, input_dim=dim))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=category, activation=tf.nn.softmax))

# 編譯模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

# 訓練模型
model.fit(x_train, y_train2,
          epochs=200,
          batch_size=64)

# 建立測試資料
x_test, y_test, y_test2 = CreateDatasets(category, 10, dim)

# 評估正確率
score = model.evaluate(x_test, y_test2, batch_size=128)
print("score:", score)

# 預測
predict = model.predict(x_test)         # 取得每一個預測結果的機率
print("predict:", predict)
y_predict = np.argmax(predict, axis=1)  # 取得每一個預測結果的答案
print("Ans:", y_predict)
print("y_test:", y_test)                # 實際的測試答案